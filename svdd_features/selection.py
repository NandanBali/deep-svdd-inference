"""Two distinct selection tracks; all arguments contain training data only."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import mutual_info_score

from .schema import candidates, feature_info
from .preprocessing import Preprocessor, valid_values, tokens

# This is a documented semantic preference, not a supervised search.
PRIORITY = """IN_BYTES OUT_BYTES IN_PKTS OUT_PKTS FLOW_DURATION_MILLISECONDS
MIN_TTL MAX_IP_PKT_LEN MIN_IP_PKT_LEN TCP_WIN_MAX_IN TCP_WIN_MAX_OUT MAX_TTL
NUM_PKTS_UP_TO_128_BYTES NUM_PKTS_128_TO_256_BYTES NUM_PKTS_256_TO_512_BYTES
NUM_PKTS_512_TO_1024_BYTES NUM_PKTS_1024_TO_1514_BYTES
LONGEST_FLOW_PKT SHORTEST_FLOW_PKT DURATION_IN DURATION_OUT
SRC_TO_DST_AVG_THROUGHPUT DST_TO_SRC_AVG_THROUGHPUT
SRC_TO_DST_IAT_MIN DST_TO_SRC_IAT_MIN SRC_TO_DST_IAT_MAX DST_TO_SRC_IAT_MAX
SRC_TO_DST_IAT_AVG DST_TO_SRC_IAT_AVG SRC_TO_DST_IAT_STDDEV DST_TO_SRC_IAT_STDDEV
RETRANSMITTED_IN_BYTES RETRANSMITTED_OUT_BYTES RETRANSMITTED_IN_PKTS RETRANSMITTED_OUT_PKTS
DNS_TTL_ANSWER PROTOCOL L4_DST_PORT L4_SRC_PORT TCP_FLAGS CLIENT_TCP_FLAGS SERVER_TCP_FLAGS
ICMP_TYPE ICMP_IPV4_TYPE L7_PROTO DNS_QUERY_TYPE FTP_COMMAND_RET_CODE""".split()


def track_a(train: pd.DataFrame, threshold=.95, modal_limit=.9995, variance_floor=1e-6):
    normal = train.loc[train.Label == 0]
    columns = candidates(list(train.columns))
    pipe = Preprocessor.fit(normal, columns)
    encoded = pipe.transform(normal)
    rows, removed, viable = [], {}, []
    for e in pipe.contract["entries"]:
        c, kind = e["raw_column"], e["kind"]
        x = valid_values(normal, c)
        counts = pd.Series(tokens(normal, c) if kind in {"port", "categorical"} else x).value_counts(dropna=False)
        mode_fraction = float(counts.iloc[0]/len(normal))
        block = encoded[:, e["offset"]:e["offset"]+len(e["output_names"])]
        variances = np.var(block.astype(np.float64), axis=0)
        reason = None
        if len(counts) <= 1:
            reason = "constant on normal training"
        elif mode_fraction >= modal_limit:
            reason = f"near-constant: normal training modal fraction >= {modal_limit}"
        elif kind == "numeric" and variances.max() < variance_floor:
            reason = f"near-constant: scaled variance < {variance_floor}"
        if reason:
            removed[c] = reason
        else:
            viable.append(c)
        rows.append(dict(feature=c, kind=kind, normal_rows=len(normal),
                         modal_fraction=mode_fraction, invalid_fraction=float(np.isnan(x).mean()),
                         encoded_variances=variances.tolist(),
                         encoded_dimension=len(e["output_names"]), removal=reason))
    numeric = [c for c in viable if feature_info(c)["kind"] == "numeric"]
    indices = [pipe.contract["output_names"].index(c) for c in numeric]
    matrix = pd.DataFrame(encoded[:, indices], columns=numeric, dtype=np.float64)
    pearson, spearman, covariance = matrix.corr(), matrix.corr(method="spearman"), matrix.cov()
    quality = {r["feature"]: r["invalid_fraction"] for r in rows}
    # Clear IP-length definitions outrank ambiguous exporter packet lengths at equal cost.
    semantic_penalty = {"LONGEST_FLOW_PKT": 1, "SHORTEST_FLOW_PKT": 1, "DURATION_IN": 1, "DURATION_OUT": 2}
    ordered = sorted(numeric, key=lambda c: (feature_info(c)["extraction_cost_ordinal"], quality[c],
        semantic_penalty.get(c, 0), PRIORITY.index(c)))
    kept, pairs = [], []
    for c in ordered:
        redundant = False
        for k in kept:
            p, s = float(pearson.loc[c, k]), float(spearman.loc[c, k])
            if max(abs(p), abs(s)) >= threshold:
                pairs.append(dict(removed=c, retained=k, pearson=p, spearman=s,
                    reason="Absolute Pearson or Spearman meets heuristic; prefer extraction cost, training data quality, explicit semantics, then declared priority"))
                removed[c] = f"normal-training redundancy with {k}"
                redundant = True
                break
        if not redundant:
            kept.append(c)
    return dict(normal_rows=len(normal), correlation_threshold=threshold,
        modal_limit=modal_limit, variance_floor=variance_floor,
        used_attack_labels=False,
        caveat="Benign filtering uses the binary label. No attack samples or attack-family labels contribute to this track. Normal variance/covariance/redundancy do not demonstrate detection usefulness; attack relationships can differ.",
        rows=rows, retained_numeric=kept,
        retained_nominal=[c for c in viable if c not in numeric], removed=removed, redundant_pairs=pairs,
        scaling=pipe.contract,
        matrices={"pearson": pearson, "spearman": spearman, "covariance": covariance})


def track_b(train: pd.DataFrame, seed=20261110, max_rows=30_000, bins=8, penalty=.1):
    rng = np.random.default_rng(seed)
    sampled = train.iloc[np.sort(rng.choice(len(train), min(len(train), max_rows), replace=False))]
    y = sampled.Label.to_numpy(dtype=int)
    if len(np.unique(y)) != 2:
        raise ValueError("Label-assisted selection needs both training classes")
    columns = candidates(list(train.columns))
    values, discrete, binned, edges, imputation = [], [], [], {}, {}
    for c in columns:
        info = feature_info(c)
        nominal = info["kind"] in {"categorical", "port", "flags"}
        is_discrete = nominal
        if nominal:
            # Flag masks are finite nominal combinations for MI, never Euclidean codes.
            t = tokens(sampled, c)
            x = pd.factorize(t, sort=True)[0].astype(float)
            z = x.astype(int)
            edges[c] = "nominal codes; ports use explicit ranges; no continuous distance"
        else:
            x = valid_values(sampled, c)
            finite = x[np.isfinite(x)]
            med = float(np.median(finite)) if len(finite) else 0.
            imputation[c] = med
            x = np.where(np.isfinite(x), x, med)
            # Low-cardinality integer measurements (notably TTL) have atoms;
            # use explicit discrete MI, avoiding a continuous jitter estimator.
            low_cardinality = len(np.unique(x)) <= 32 and np.all(x == np.floor(x))
            if info["log1p"]:
                x = np.log1p(x)
            if low_cardinality:
                is_discrete = True
                x = pd.factorize(x, sort=True)[0].astype(float)
                z = x.astype(int)
                edges[c] = "exact codes for <=32 distinct integer measurements in estimation sample"
            else:
                limits = np.unique(np.quantile(x, np.linspace(0, 1, bins+1)[1:-1]))
                # Values equal to a tied cutpoint stay in the lower bin, preserving
                # rare upper tails when most observations share the same value.
                z = np.searchsorted(limits, x, side="left")
                edges[c] = limits.tolist()
        values.append(x)
        binned.append(z)
        discrete.append(is_discrete)
    X, Z = np.column_stack(values), np.column_stack(binned)
    relevance = mutual_info_classif(X, y, discrete_features=np.array(discrete),
                                    random_state=seed, n_neighbors=3)
    relevance_bin = np.array([mutual_info_score(Z[:, i], y) for i in range(len(columns))])
    ranking = sorted([dict(feature=c, mi_nats=float(relevance[i]),
        binned_mi_nats=float(relevance_bin[i]), discrete=discrete[i],
        extraction_cost_ordinal=feature_info(c)["extraction_cost_ordinal"]) for i, c in enumerate(columns)],
        key=lambda r: (-r["mi_nats"], r["feature"]))
    # Greedy mRMR-style difference criterion; calculate only selected-to-pool pairs.
    remaining = [i for i in range(len(columns)) if np.unique(Z[:, i]).size > 1]
    selected, steps, redundancy = [], [], {}
    encoding = Preprocessor.fit(train.loc[train.Label == 0], columns)
    dims = {e["raw_column"]: len(e["output_names"]) for e in encoding.contract["entries"]}
    used_dim = 0
    while remaining and len(selected) < 10:
        options = []
        for i in remaining:
            if used_dim+dims[columns[i]] > 16:
                continue
            for j in selected:
                if (i, j) not in redundancy:
                    redundancy[i, j] = float(mutual_info_score(Z[:, i], Z[:, j]))
            red = float(np.mean([redundancy[i, j] for j in selected])) if selected else 0.
            objective = relevance_bin[i]-penalty*red
            options.append((float(objective), i, red))
        if not options:
            break
        objective, i, red = max(options, key=lambda t: (t[0], -feature_info(columns[t[1]])["extraction_cost_ordinal"], -t[1]))
        if objective <= 0:
            break
        selected.append(i)
        remaining.remove(i)
        used_dim += dims[columns[i]]
        steps.append(dict(feature=columns[i], relevance_nats=float(relevance_bin[i]),
                          mean_redundancy_nats=red, objective=objective, encoded_dimension=used_dim))
    return dict(uses_attack_labels=True, rows=len(sampled), seed=seed,
        class_distribution={str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
        sampling="Uniform without replacement from retained training rows; no balancing. Split and sampling/purging change prevalence relative to full CSV.",
        mi_estimator="sklearn mutual_info_classif: 3-neighbor continuous/discrete estimator; discrete mask explicit; nominal MI for flags, protocol and port ranges; exact discrete MI for numeric measurements with <=32 unique integer values in training estimation sample; remaining numeric measurements continuous, with log1p where audited; reproducible seed; nats",
        mrmr_formula="J(f)=I_bin(f;binary_label) - lambda * mean_{s in selected} I_bin(f;s); first term only when selected empty; lambda=penalty; stop at J<=0, 10 raw features or 16 encoded inputs",
        redundancy_estimator="Uncorrected empirical contingency-table mutual information (sklearn mutual_info_score), nats; numeric features discretized into at most 8 training-quantile bins (ties assigned lower bin), except <=32 unique integer values use exact codes; nominal features exact codes/ranges. Same discretization used for relevance in mRMR. Finite-sample bias; no claim of optimal subset.",
        bins=bins, penalty=penalty, bin_edges=edges, medians=imputation,
        ranking=ranking, mrmr_selected=[columns[i] for i in selected], mrmr_steps=steps)


def make_candidates(train: pd.DataFrame, a: dict, b: dict):
    normal = train.loc[train.Label == 0]
    # A: bounded, predeclared semantic priority, no validation or attack relevance.
    eligible = [c for c in PRIORITY if c in a["retained_numeric"] and feature_info(c)["extraction_cost_ordinal"] <= 2]
    compact = eligible[:8]
    if not compact:
        raise ValueError("No usable hardware-friendly features")
    balanced = compact.copy()
    for c in ["PROTOCOL", "L4_DST_PORT"]:
        if c in balanced:
            continue
        if c not in a["retained_numeric"] + a["retained_nominal"]:
            continue
        if feature_info(c)["extraction_cost_ordinal"] > 3:
            continue
        proposed = balanced+[c]
        if Preprocessor.fit(normal, proposed).contract["input_dimension"] <= 16:
            balanced = proposed
        if Preprocessor.fit(normal, balanced).contract["input_dimension"] >= 16:
            break
    return {
        "A_compact": dict(track="A", features=compact,
                          rationale="First up to eight retained cost<=2 measurements by declared semantic priority; minimum FPGA extraction/preprocessing scope."),
        "A_balanced": dict(track="A", features=balanced,
                           rationale="Compact plus nominal protocol and destination port-range context when retained; bounded to 16 encoded inputs, no padding."),
        "B_mrmr": dict(track="B (uses attack training labels)", features=b["mrmr_selected"],
                       rationale="Greedy empirical quantile-bin mRMR difference criterion; includes attack-label assistance and can require expensive extraction."),
    }
