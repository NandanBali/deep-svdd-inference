"""Audited schema for the downloaded release. Fail closed on unfamiliar columns.

Meanings below are paraphrased; engineering assessments are our estimates, not
synthesis results. Sources and remaining exporter ambiguities are explicit.
"""
from __future__ import annotations

from copy import deepcopy

SOURCES = {
    "dataset": "https://rdm.uq.edu.au/files/abd2f5d8-e268-4ff0-84fb-f2f7b3ca3e8f",
    "feature_file": "sources/NetFlow_v3_Features.csv (verified archive payload)",
    "paper": "https://arxiv.org/html/2503.04404v3#S3.T1",
    "nprobe": "https://www.ntop.org/guides/nprobe/cli_options.html",
    "nprobe_elements": "https://www.ntop.org/guides/nprobe/flow_information_elements.html",
    "seconds": "https://ntop.org/introducing-per-second-measurements-in-nprobe-flow-exports/",
}

_REGISTRY = {}


def add(names, meaning, kind, units, source, state, operations, cost, explanation,
        *, direction="both directions aggregated", unknown="", candidate=True,
        log=False, maximum=None, payload=False, availability="flow export/expiry"):
    for name in names.split():
        _REGISTRY[name] = dict(
            name=name, meaning=meaning, kind=kind, units=units, source=source,
            per_flow_state=state, update_operations=operations,
            extraction_cost_ordinal=cost, extraction_cost_explanation=explanation,
            direction=direction, availability=availability,
            completed_flow_required=availability == "flow export/expiry",
            cross_flow_history=False, payload_inspection=payload,
            unknowns=unknown or "Verify nProbe version, expiry policy and packet-count/byte conventions against PCAP before FPGA extraction equivalence.",
            candidate=candidate, log1p=log, max_value=maximum,
            preprocessing_cost=("software log1p + clipping + saved affine map" if log else
                                "clipping + saved affine map" if kind == "numeric" else
                                "bit unpacking" if kind == "flags" else
                                "category comparisons and one-hot" if kind in {"categorical", "port"} else "excluded"),
            neural_input_cost=("1 scalar" if kind == "numeric" else "8 bits" if kind == "flags" else
                               "at most 4 one-hot values; port ranges use 5 incl other" if kind in {"categorical", "port"} else "0"),
            sources=[SOURCES["feature_file"], SOURCES["paper"], SOURCES["nprobe"]],
        )


add("IPV4_SRC_ADDR IPV4_DST_ADDR", "Flow endpoint address", "address", "IPv4 address", "packet header",
    "flow key", "header extraction and flow-table lookup", 1, "Header field, but storing/looking up flow keys has shared table cost.",
    candidate=False, availability="first packet", unknown="Raw addresses excluded: identity/capture shortcut; IPs remain metadata only.")
add("FLOW_START_MILLISECONDS FLOW_END_MILLISECONDS", "First/last observed packet time", "timestamp", "Unix epoch milliseconds", "timestamp",
    "first and last timestamp", "latch first; update last", 2, "Timestamp acquisition and state; excluded from MLP.",
    candidate=False, unknown="Used for chronological separation, not model input. Verify clock precision and exporter time policy.")
add("Label Attack", "Binary ground truth / attack-family annotation", "label", "0 benign, 1 attack / category string", "ground truth",
    "none", "none", 1, "Annotation, not an extractable inference feature.", candidate=False,
    unknown="Never input to preprocessing. Track B explicitly uses Label for selection.")
add("L4_SRC_PORT L4_DST_PORT", "Transport endpoint port, interpreted as a service/range code", "port", "port code 0..65535", "packet header",
    "16-bit field in flow key", "latch header; range comparisons for encoding", 1, "Simple header field; range encoding adds comparisons, not numeric distance.",
    maximum=65535, availability="first packet", direction="source endpoint / destination endpoint respectively",
    unknown="Port 0 can denote absent/not-applicable. Fixed ranges discard exact service identity; no assumption of server port or ephemeral client direction.")
add("PROTOCOL", "IP transport protocol code", "categorical", "IANA protocol code", "packet header",
    "8-bit field in flow key", "latch header", 1, "Header byte; categorical lookup is preprocessing.", maximum=255,
    availability="first packet", direction="flow-wide",
    unknown="Numeric codes are nominal. Known categories learned on normal training; remaining codes map to other.")
add("L7_PROTO", "nProbe application classification code", "categorical", "nominal exporter code (possibly master.application)", "application parsing",
    "classifier state; potentially payload and multi-packet context", "protocol-specific parsing / classification", 5,
    "Application classification requires substantially more than a header byte.", payload=True, direction="flow-wide",
    unknown="Exporter/nDPI version and decimal code semantics unrecorded; preserve as nominal token, never round or scale as magnitude.")
for names, unit in [("IN_BYTES OUT_BYTES", "bytes"), ("IN_PKTS OUT_PKTS", "packets")]:
    add(names, "Directional flow traffic total", "numeric", unit, "per-flow counter",
        "one counter per direction", "add packet length / increment", 2, "Counter read-modify-write per packet plus flow-table access.",
        log=True, direction="IN: src->dst; OUT: dst->src",
        unknown="IN/OUT are exporter directions, not LAN ingress/egress. Byte layer, overflow and active-expiry policies require PCAP equivalence checks.")
add("TCP_FLAGS CLIENT_TCP_FLAGS SERVER_TCP_FLAGS", "TCP control bits accumulated by bitwise OR", "flags", "8-bit mask", "packet header",
    "one mask (or one per direction)", "bitwise OR", 1, "Small mask state; MLP cost is eight bit inputs per original field.", maximum=255,
    direction="TCP_FLAGS both; CLIENT/SERVER exporter client/server orientation",
    unknown="Preserve all eight bits; verify client/server orientation, non-TCP zero, ECN and malformed-packet behavior.")
add("FLOW_DURATION_MILLISECONDS", "Elapsed lifetime of exported flow", "numeric", "milliseconds", "timestamp",
    "first and last timestamp", "timestamp subtraction at export", 2, "Two timestamps shared with flow expiry; one subtraction.", log=True,
    unknown="CSV duration need not equal end-start exactly (rounding/exporter precision). Keep supplied value; no silent replacement.")
add("DURATION_IN DURATION_OUT", "Exporter stream duration by direction (reverse definition ambiguous)", "numeric", "milliseconds", "timestamp",
    "first/last packet times per direction", "timestamp updates and subtraction", 2, "Directional timestamps, shared with IAT if retained.", log=True,
    direction="IN presumed src->dst; OUT presumed dst->src, not verified",
    unknown="Archive and nProbe CLI table describe OUT as client->server; nProbe information-elements table instead says server->client. The latter is the likely reverse direction, but the dataset exporter/zero-packet convention still needs verification; excluded from compact recommendation.")
add("MIN_TTL MAX_TTL", "Smallest/largest observed IP time-to-live field", "numeric", "hop-limit value 0..255", "packet header",
    "one extremum per field", "comparison and conditional update", 1, "Small header comparison and byte of state per extremum.", maximum=255,
    unknown="TTL can encode host/topology/capture identity. Needs cross-network robustness check; not elapsed time.")
add("LONGEST_FLOW_PKT SHORTEST_FLOW_PKT", "Largest/smallest exporter packet length", "numeric", "bytes", "packet header",
    "packet-size extrema", "comparison and conditional update", 1, "Extrema compare a packet length; no division.",
    unknown="Packet layer/headers included and relationship to IP length extrema are not explicit. Prefer explicitly IP-named measurements when redundant.")
add("MIN_IP_PKT_LEN MAX_IP_PKT_LEN", "Smallest/largest observed IP packet length", "numeric", "bytes", "packet header",
    "IP length extrema", "comparison and conditional update", 1, "Header length comparison and extrema state.",
    unknown="Verify IPv4 total-length, fragments, empty-direction sentinel and packet observation policy.")
add("SRC_TO_DST_SECOND_BYTES DST_TO_SRC_SECOND_BYTES", "Exporter per-second byte measurement reduced to a scalar in this CSV", "numeric", "ambiguous per-second byte representation", "derived statistic",
    "time buckets and directional byte totals", "time-bucket comparison; accumulation; exporter-specific reduction", 4,
    "nProbe documents per-second series; the dataset scalar reduction is undocumented.", candidate=False, log=True,
    direction="name denotes src->dst or dst->src; verify dataset conversion",
    unknown="Not simply total bytes/duration. Rejected for unresolved CSV reduction semantics; do not implement an approximation.")
add("RETRANSMITTED_IN_BYTES RETRANSMITTED_OUT_BYTES RETRANSMITTED_IN_PKTS RETRANSMITTED_OUT_PKTS",
    "TCP retransmission volume by exporter direction", "numeric", "bytes for *_BYTES; packets for *_PKTS", "derived statistic",
    "per-direction sequence tracking, counters, possible sequence intervals", "TCP sequence comparisons, wrap handling and additions", 4,
    "Retransmission identification requires TCP sequence history; counters alone are insufficient.", log=True,
    direction="IN src->dst; OUT dst->src",
    unknown="Exporter retransmission/out-of-order heuristic, partial overlaps and sequence-wrap policy unknown.")
add("SRC_TO_DST_AVG_THROUGHPUT DST_TO_SRC_AVG_THROUGHPUT", "Exporter mean directional bit rate", "numeric", "bits/second", "derived statistic",
    "byte total and duration; exporter rate state", "multiplication by unit factor and division", 3,
    "Division at export can share a reciprocal; feasible in software v1, more involved than a counter.", log=True,
    direction="as named",
    unknown="Exact denominator, rounding, zero-duration and saturation conventions require verification. Assessed individually; not rejected merely for division.")
add("NUM_PKTS_UP_TO_128_BYTES NUM_PKTS_128_TO_256_BYTES NUM_PKTS_256_TO_512_BYTES NUM_PKTS_512_TO_1024_BYTES NUM_PKTS_1024_TO_1514_BYTES",
    "Flow packet count in the named IP-length interval", "numeric", "packets", "per-flow counter",
    "one bin counter per retained interval", "IP length comparisons and conditional increment", 2,
    "Several comparisons can share a size decoder; counters still require per-flow updates.", log=True,
    unknown="Use archive boundaries: first <=128; other bins lower-exclusive, upper-inclusive. nProbe documentation differs on some upper bounds; >1514 packets absent from this bin set.")
add("TCP_WIN_MAX_IN TCP_WIN_MAX_OUT", "Maximum observed directional TCP advertised window", "numeric", "TCP window field units", "packet header",
    "maximum window per direction", "comparison and conditional update", 1,
    "Header-field maximum; verify unscaled versus negotiated effective window before extraction.",
    direction="IN src->dst; OUT dst->src",
    unknown="Window scaling, non-TCP zero and negotiated-window interpretation need exporter confirmation; dataset alone cannot resolve them.")
add("ICMP_TYPE ICMP_IPV4_TYPE", "ICMP type/code or type annotation", "categorical", "ICMP_TYPE=type*256+code; ICMP_IPV4_TYPE=type", "packet header",
    "type and code", "header extraction and concatenation", 1, "Header codes; categorical encoding avoids ordinal distances.",
    maximum=65535, direction="packet direction captured by exporter",
    unknown="Non-ICMP zero can collide with valid type/code; verify last/first/aggregate policy for multi-packet flows.")
add("DNS_QUERY_ID", "DNS transaction identifier", "identifier", "16-bit transaction code", "application parsing",
    "DNS parser and transaction identifier", "DNS message parsing", 4, "Application header parsing; identifier lacks useful magnitude.",
    candidate=False, payload=True, unknown="Rejected as transaction identifier / possible shortcut, not a continuous measurement.")
add("DNS_QUERY_TYPE", "DNS record query class by type", "categorical", "DNS type code", "application parsing",
    "DNS parser; selected query type", "DNS header/question parsing", 4,
    "Requires application payload access and DNS message parsing beyond transport headers.", payload=True,
    unknown="First/last query policy, compressed names, TCP DNS reassembly and absent-message sentinel unknown.")
add("DNS_TTL_ANSWER", "TTL attached to exporter-selected DNS A answer", "numeric", "seconds", "application parsing",
    "DNS answer parser, selected TTL", "variable-length parsing; possible name compression and TCP reassembly", 5,
    "Payload parsing and record selection dominate extraction; affine scaling is small by comparison.", log=True, payload=True,
    unknown="First A-record policy, zero sentinel and multi-answer handling require exporter matching; not IP TTL.")
add("FTP_COMMAND_RET_CODE", "FTP reply status code observed by exporter", "categorical", "nominal reply code", "application parsing",
    "FTP control-stream parser and reply state", "ASCII parsing; possible TCP reassembly", 5,
    "Application parsing and stream state; code itself is small.", payload=True,
    unknown="Dataset wording says client return code; reply direction and selection rule ambiguous. Encrypted FTP unavailable without decryption.")
for direction in ["SRC_TO_DST", "DST_TO_SRC"]:
    for stat in ["MIN", "MAX", "AVG", "STDDEV"]:
        cost = 5 if stat == "STDDEV" else 4 if stat == "AVG" else 3
        state = "previous directional timestamp; " + ("sum, squared sum (or stable moments), count" if stat == "STDDEV" else
            "sum and count" if stat == "AVG" else "gap extremum")
        ops = "timestamp subtraction; " + ("moment updates, multiplication, division and square root" if stat == "STDDEV" else
            "accumulation and division at export" if stat == "AVG" else "comparison")
        add(f"{direction}_IAT_{stat}", f"Directional inter-packet gap {stat.lower()}", "numeric", "milliseconds per nProbe documentation", "derived statistic",
            state, ops, cost, "Per-direction gap tracking; arithmetic cost depends on the specific statistic.", log=True,
            direction=direction.lower().replace("_", " "),
            unknown="Archive omits IAT units and repeats incorrect reverse-direction descriptions. nProbe documents msec; version, sample/population stddev and <2-packet sentinel require verification.")


def feature_info(column: str) -> dict:
    if column not in _REGISTRY:
        raise ValueError(f"Unaudited column: {column}")
    return deepcopy(_REGISTRY[column])


def audit(columns: list[str]) -> list[dict]:
    return [feature_info(c) for c in columns]


def expected_columns() -> set[str]:
    return set(_REGISTRY)


def candidates(columns: list[str]) -> list[str]:
    return [c for c in columns if feature_info(c)["candidate"]]
