# Feature-selection handoff

Historical feature-selection report (2026-11-10). The subsequent trained software
baseline requested in `prompt-software.md` is documented in `SOFTWARE.md` and its
generated `runs/software/report.md`. Baseline-unavailable statements below describe
the state at the time of this earlier study.

**Recommend `A_compact`: eight original features, eight encoded inputs**, in this
order:

```text
IN_BYTES, OUT_BYTES, FLOW_DURATION_MILLISECONDS, MIN_TTL,
MAX_IP_PKT_LEN, MIN_IP_PKT_LEN, TCP_WIN_MAX_IN, TCP_WIN_MAX_OUT
```

This is a **provisional engineering recommendation** for the two-week Pynq-Z2
inference-core milestone. It provides traffic volume, lifetime, packet-length and
TCP-window measurements with counters, extrema and timestamps. It requires no
application parser, retransmission tracker, divider or cross-flow history for
extraction. No existing Deep SVDD evaluator was available, so detection performance
has not been demonstrated. The independently developed baseline must validate it.

## Actual data and separation

Downloaded the official [UQ NF-UNSW-NB15-v3 release](https://rdm.uq.edu.au/files/abd2f5d8-e268-4ff0-84fb-f2f7b3ca3e8f),
DOI `10.48610/6E0EDA1`, and verified the archive's BagIt payload checksums. The
actual CSV has 53 raw features plus `Label` and `Attack`: **2,365,424 flows**,
**2,237,731 benign**, **127,693 attacks**. `Label=0` matches `Attack=Benign` for
every row. Nine attack families are present. Identity/checksums are in
[sources/provenance.json](sources/provenance.json); all column types, ranges,
missing values and distributions are in [results/inspection.json](results/inspection.json).

The file's family counts differ from the [dataset website's table](https://staff.itee.uq.edu.au/marius/NIDS_datasets/):
the CSV contains Analysis **1,226**, Backdoor **4,659**, Shellcode **2,381**.
The website lists different counts for those names. Its listed attack total also
differs from the file. The workflow preserves the verified CSV labels, checks their
binary consistency and records the disagreement; it does not relabel families.

Full-file checks found **14,815 duplicate rows**, **230,815 repeated eligible raw
feature vectors**, no conflicting binary labels for those raw vectors, no negative
numeric values and no globally constant numeric columns. `SRC_TO_DST_SECOND_BYTES`
contains **63,425 missing** and **59,068 infinite** values;
`DST_TO_SRC_SECOND_BYTES` contains **122,493 infinite** values. No other column has
missing/infinite values. Both fields are excluded: their scalar meaning is also
unclear relative to nProbe's [per-second series documentation](https://ntop.org/introducing-per-second-measurements-in-nprobe-flow-exports/).
No substitute rate was invented. `L7_PROTO` includes **23,866 fractional codes**;
they remain nominal tokens, not continuous magnitudes.

Splits precede all fitting: 60%/20%/20% chronological start-time cutoffs, a 60-second
embargo on each side of each boundary, and removal of flows whose end crosses the
allowed boundary. Later-partition repetitions of eligible raw vectors are purged;
within-partition multiplicity remains. This removes **7,253** time/boundary rows
and **120,705** later duplicate occurrences. Remaining train/validation/test sizes
are **1,417,416 / 411,229 / 408,821**. No IP, timestamp, label, transaction ID, row ID
or source/split marker enters the model. No row-ID/source/split column exists in
this schema.

Each partition is sampled uniformly without balancing to at most **100,000** rows,
seed **20261110**. Preprocessing and Track A use **97,010 normal training rows**.
Training spans January 22 through early February 18, 2015; validation and test use
successive later periods on February 18. Same hosts/capture can recur across
periods. This is temporal separation, not independent-network/capture validation.

Removing features or clipping can make previously distinct vectors identical.
After these checks, **33,310 identical validation row IDs** remain for all three
candidates; the frozen compact choice has **39,166 test rows** exported without
scoring. These are substantially filtered cohorts. Their class prevalence and
future PR-AUC/FPR/recall must not be presented as unfiltered traffic-population
performance. See [results/findings.md](results/findings.md) for purge counts.

## Separate selection tracks and candidates

**Track A** uses only known-normal training samples. It removes near-constant
measurements and uses scaled covariance plus Pearson/Spearman redundancy, with
absolute correlation **0.95** as a heuristic. Pair selection prefers ordinal
extraction cost, training data quality, clearer definitions, then a declared
semantic priority. Examples: `LONGEST_FLOW_PKT` and `MAX_IP_PKT_LEN` correlate
exactly; `MAX_TTL` versus `MIN_TTL` has Pearson **0.9914**; packet counts versus
`TCP_WIN_MAX_IN` have Spearman **0.9683 / 0.9830**. The latter are empirical benign
relationships, not physical equivalences: rejecting packet counts may lose attack
information. Covariance/variance only show variability and redundancy; they are
not evidence of anomaly-detection usefulness.

**Track B uses attack training labels.** It estimates MI on **30,000** unbalanced
training rows (**29,091 benign / 909 attacks**). Nominal values, port ranges, masks
and low-cardinality integer measurements use explicit discrete estimation;
remaining measurements use the seeded 3-neighbor continuous estimator. Highest MI
includes MIN_TTL **0.1350**, MAX_TTL **0.1348**, SHORTEST_FLOW_PKT **0.1320** and
MIN_IP_PKT_LEN **0.1255** nats. TTL may be a strong host/topology/testbed shortcut.
Raw relevance is not a Deep SVDD result or proof that clipped/encoded inputs retain
the same information.

The greedy mRMR-style objective is `I_bin(f;Label) - 0.1 * mean I_bin(f;s)` over
already selected features. Relevance and redundancy in that formula use empirical
contingency-table MI in nats, with up to eight training-quantile bins for continuous
measurements, exact codes for low-cardinality integer measurements and nominal
fields. Stop at nonpositive gain, ten raw features or sixteen encoded inputs.
This is distinct from the separately exported MI ranking. Bin boundaries, sample
seed, estimator caveats, every gain and redundancy term are in
[results/track_b.json](results/track_b.json).

| Candidate | Track | Original / encoded | Retained features and tradeoff |
|---|---|---:|---|
| A_compact | Normal-only A | 8 / 8 | Recommended measurements above; small extraction state and input dimension. |
| A_balanced | Normal-only A | 10 / 15 | Compact plus PROTOCOL (three normal-learned codes, unknown all-zero) and L4_DST_PORT (four port ranges); more protocol/service context without ordinal code distances. |
| B_mrmr | Label-assisted B | 6 / 16 | MIN_TTL, MAX_TTL, MIN_IP_PKT_LEN, SERVER_TCP_FLAGS (8 bits), DST_TO_SRC_AVG_THROUGHPUT, L7_PROTO (4 one-hot inputs). Adds control flags, a directional rate and application context, but extraction is substantially more involved. |

No attack validation labels chose these subsets or the recommendation. No test
scores/metrics were computed. Only training statistics, predefined budgets and
engineering scope inform the provisional recommendation.

## Contributions and implementation cost

These contributions are hypotheses to validate with the baseline. Ordinal costs
are **engineering assessments**, not synthesis measurements; shared flow-table
lookup/update and expiry state must also be budgeted.

| Retained compact feature | Intended information | Extraction and state | Ordinal cost |
|---|---|---|---:|
| IN_BYTES | Source-to-destination traffic volume | Add packet bytes to forward counter | 2 |
| OUT_BYTES | Response volume/asymmetry relative to forward bytes | Add packet bytes to reverse counter | 2 |
| FLOW_DURATION_MILLISECONDS | Lifetime distinct from traffic volume | Shared first/last timestamps; subtraction at export | 2 |
| MIN_TTL | Minimum observed hop-limit field; topology signature risk | Compare IP header TTL with saved minimum | 1 |
| MAX_IP_PKT_LEN | Largest IP packet / transfer-size context | Length comparison and maximum state | 1 |
| MIN_IP_PKT_LEN | Small-packet/control traffic context | Length comparison and minimum state | 1 |
| TCP_WIN_MAX_IN | Forward TCP receive-window behavior | TCP header comparison and maximum state | 1 |
| TCP_WIN_MAX_OUT | Reverse TCP receive-window behavior | Separate reverse maximum | 1 |

`A_balanced` adds a latched IP protocol code and destination port-range comparisons
(ordinal extraction cost 1 each). Exact port identities are deliberately discarded;
this can lose service-specific detection information. `B_mrmr` adds MAX_TTL (a
maximum comparator, 1), SERVER_TCP_FLAGS (OR mask, 1; eight neural inputs), reverse
mean throughput (rate state/division, 3) and L7_PROTO (application classifier,
potential payload/multi-packet state, 5). A rate is feasible with software or a
shared reciprocal; it is not rejected simply because it divides. The application
classifier is a poor first extraction target for the two-week milestone.

Important exclusions: raw IPs and DNS_QUERY_ID are identifiers; labels and absolute
times are metadata. Packet counts and extra duration fields are removed by the
normal redundancy heuristic. Directional IAT minima are nearly constant (the
reverse minimum is removed); averages/stddev need additional timestamp/moment
state, division and sometimes square root. Retransmission features need sequence
history and overlap/wrap rules. DNS/FTP fields need application parsing. None of
these is assumed cheap merely because its CSV representation is numeric.

V1 preprocessing stays in software: training medians, three log1p transforms,
saved affine constants and float32 output. Normal q0.001/q0.999 are scale anchors;
saturation extends the numeric output range to **[-1,4]**. Strict clipping at the
normal upper quantile would map both TTL 31 and TTL 252 to the same value; bounded
extension preserves that contrast. Far-tail distinctions still saturate. The
contract fixes missing, invalid and unseen-value handling and never refits on
validation/test. Extraction cost, preprocessing cost and neural input dimension
are separate audit fields.

All aggregate inputs describe the exported/expired flow, not an equivalent early
packet decision. Before FPGA extraction, verify byte-counting layer, direction,
window scaling, timestamps and active/idle expiry against the exact exporter and
PCAPs. `DURATION_OUT` and several bundled reverse-IAT descriptions are ambiguous;
IAT milliseconds come from [nProbe's CLI definitions](https://www.ntop.org/guides/nprobe/cli_options.html),
not the incomplete bundled unit descriptions. No hardware approximations or
LUT/DSP/BRAM/throughput estimates are supplied without equivalence tests/synthesis.

## Baseline status, verification and exact handoff

ROC-AUC, PR-AUC, measured 1%-calibration-point recall/FPR, per-attack recall, model
score distributions and embedding-collapse results are **unavailable**. The
integration API implements those calculations for the future baseline. Compare
all three candidates with the same bias-free `d -> 16 -> 8` policy, budget and
predeclared seeds, with separate centers/thresholds. Calibrate on normal validation
scores; disclose any attack validation labels used for selection; freeze the final
choice before a single final test evaluation per seed. Some rare attack families
have very few retained validation examples, so their recall estimates will be
unstable even after the baseline exists.

`./reproduce.sh` completed, including **10 regression tests**. The run verified
save/reload equality, stable column ordering, label exclusion, finite float32
outputs, raw/exported golden-vector agreement and encoded partition separation.
Golden cases contain four normal and one example per attack family from training.
A second full run reproduced both tracks, the selected contracts and all exported
array hashes exactly.

- **Baseline developer:** [results/selection.json](results/selection.json), each
  candidate's `preprocessing.json` and `baseline.npz`, and the score-interface
  instructions in [README.md](README.md).
- **FPGA developer:** [results/A_compact/preprocessing.json](results/A_compact/preprocessing.json),
  [example_raw.csv](results/A_compact/example_raw.csv),
  [example_vectors.csv](results/A_compact/example_vectors.csv),
  [example_labels.csv](results/A_compact/example_labels.csv), and
  [results/feature_audit.json](results/feature_audit.json).

The baseline must still supply trained weights, center, activation details,
threshold and a hardware numeric-precision agreement. Until its validation is
available, **use the eight-input compact contract as the provisional milestone
interface**.
