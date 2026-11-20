# Generated run findings

Recommendation: **A_compact (provisional; baseline absent)**.

Full CSV: 2,365,424 flows, 53 raw features, 2 labels; SHA-256 `4ebb97bd74412d566137d95a6fc3ffd8f374f1cf8cfe204d007848e7a668f9b5`.
Exact duplicate rows (hash-based): 14,815; duplicate eligible raw feature vectors: 230,815; conflicting-label feature groups: 0.

## Full-file family distribution

| Family | Rows |
|---|---:|
| Analysis | 1,226 |
| Backdoor | 4,659 |
| Benign | 2,237,731 |
| DoS | 5,980 |
| Exploits | 42,748 |
| Fuzzers | 33,816 |
| Generic | 19,651 |
| Reconnaissance | 17,074 |
| Shellcode | 2,381 |
| Worms | 158 |

## Split and sampling

Cutoffs (epoch ms): [1424229622862, 1424246259997]; symmetric embargo: 60000 ms.
Time/boundary purges: 7,253; cross-partition raw-vector duplicate purges: 120,705.

| Partition | Eligible rows | Uniform bounded sample | Normal / attack in eligible partition |
|---|---:|---:|---|
| train | 1,417,416 | 100,000 | {'0': 1375035, '1': 42381} |
| validation | 411,229 | 100,000 | {'0': 373424, '1': 37805} |
| test | 408,821 | 100,000 | {'0': 372824, '1': 35997} |

## Track A: no attack samples for ranking

97,010 normal training rows; absolute Pearson OR Spearman threshold 0.95; modal limit 0.9995. Covariance uses training-fitted log/quantile/min-max scaling.
Normal variability/redundancy establishes distinguishable inputs, not anomaly-detection performance.

Removed inputs:
- `DST_TO_SRC_IAT_MIN`: near-constant: normal training modal fraction >= 0.9995.
- `MAX_TTL`: normal-training redundancy with MIN_TTL.
- `LONGEST_FLOW_PKT`: normal-training redundancy with MAX_IP_PKT_LEN.
- `IN_PKTS`: normal-training redundancy with TCP_WIN_MAX_IN.
- `OUT_PKTS`: normal-training redundancy with TCP_WIN_MAX_IN.
- `NUM_PKTS_UP_TO_128_BYTES`: normal-training redundancy with TCP_WIN_MAX_IN.
- `DURATION_IN`: normal-training redundancy with FLOW_DURATION_MILLISECONDS.
- `DURATION_OUT`: normal-training redundancy with FLOW_DURATION_MILLISECONDS.
- `SRC_TO_DST_IAT_MAX`: normal-training redundancy with FLOW_DURATION_MILLISECONDS.
- `DST_TO_SRC_IAT_MAX`: normal-training redundancy with FLOW_DURATION_MILLISECONDS.
- `DST_TO_SRC_IAT_AVG`: normal-training redundancy with SRC_TO_DST_IAT_AVG.
- `RETRANSMITTED_OUT_BYTES`: normal-training redundancy with OUT_BYTES.
- `RETRANSMITTED_IN_PKTS`: normal-training redundancy with TCP_WIN_MAX_OUT.
- `RETRANSMITTED_OUT_PKTS`: normal-training redundancy with TCP_WIN_MAX_IN.
- `SRC_TO_DST_IAT_STDDEV`: normal-training redundancy with SRC_TO_DST_IAT_AVG.
- `DST_TO_SRC_IAT_STDDEV`: normal-training redundancy with SRC_TO_DST_IAT_AVG.

## Track B: uses attack training labels

MI sample: 30,000 training rows without balancing; class counts {'0': 29091, '1': 909}; seed 20261110.
sklearn mutual_info_classif: 3-neighbor continuous/discrete estimator; discrete mask explicit; nominal MI for flags, protocol and port ranges; exact discrete MI for numeric measurements with <=32 unique integer values in training estimation sample; remaining numeric measurements continuous, with log1p where audited; reproducible seed; nats.
J(f)=I_bin(f;binary_label) - lambda * mean_{s in selected} I_bin(f;s); first term only when selected empty; lambda=penalty; stop at J<=0, 10 raw features or 16 encoded inputs.
Uncorrected empirical contingency-table mutual information (sklearn mutual_info_score), nats; numeric features discretized into at most 8 training-quantile bins (ties assigned lower bin), except <=32 unique integer values use exact codes; nominal features exact codes/ranges. Same discretization used for relevance in mRMR. Finite-sample bias; no claim of optimal subset.

| Feature | MI (nats) | Quantile-bin relevance (nats) |
|---|---:|---:|
| MIN_TTL | 0.135024 | 0.135024 |
| MAX_TTL | 0.134809 | 0.134809 |
| SHORTEST_FLOW_PKT | 0.131993 | 0.001591 |
| MIN_IP_PKT_LEN | 0.125468 | 0.125468 |
| TCP_WIN_MAX_OUT | 0.097349 | 0.048021 |
| OUT_BYTES | 0.096901 | 0.015737 |
| MAX_IP_PKT_LEN | 0.095661 | 0.010266 |
| LONGEST_FLOW_PKT | 0.094153 | 0.010266 |
| IN_BYTES | 0.093392 | 0.011983 |
| TCP_WIN_MAX_IN | 0.092146 | 0.046937 |
| RETRANSMITTED_IN_BYTES | 0.074237 | 0.012461 |
| DST_TO_SRC_AVG_THROUGHPUT | 0.071436 | 0.052458 |
| SERVER_TCP_FLAGS | 0.067102 | 0.067102 |
| TCP_FLAGS | 0.063705 | 0.063705 |
| NUM_PKTS_UP_TO_128_BYTES | 0.061498 | 0.017325 |

## Three bounded candidates

| Candidate | Track | Raw | Encoded | Features |
|---|---|---:|---:|---|
| A_compact | A | 8 | 8 | IN_BYTES, OUT_BYTES, FLOW_DURATION_MILLISECONDS, MIN_TTL, MAX_IP_PKT_LEN, MIN_IP_PKT_LEN, TCP_WIN_MAX_IN, TCP_WIN_MAX_OUT |
| A_balanced | A | 10 | 15 | IN_BYTES, OUT_BYTES, FLOW_DURATION_MILLISECONDS, MIN_TTL, MAX_IP_PKT_LEN, MIN_IP_PKT_LEN, TCP_WIN_MAX_IN, TCP_WIN_MAX_OUT, PROTOCOL, L4_DST_PORT |
| B_mrmr | B (uses attack training labels) | 6 | 16 | MIN_TTL, MAX_TTL, MIN_IP_PKT_LEN, SERVER_TCP_FLAGS, DST_TO_SRC_AVG_THROUGHPUT, L7_PROTO |

## Baseline status and freeze

No working Deep SVDD evaluator or model existed in the repository. No neural training was added. ROC-AUC, PR-AUC, recall/FPR, per-attack recall, score distributions and embedding-collapse results are **not available**.
No attack validation labels were used to choose the recommendation. The provisional choice was saved before test raw features were loaded for the handoff export. No test scores or metrics were computed.
After subset selection, clipping and encoding can create new duplicate vectors. Validation collisions against all mixed training inputs were removed per candidate, then intersected into 33,310 identical validation row IDs for fair comparison. This substantially changes the population. Test collisions were purged against training/retained validation for the frozen recommendation only; other test arrays are not exported.

| Candidate | Validation vectors purged | Test vectors purged | Test rows exported |
|---|---:|---:|---:|
| A_compact | 58,451 | 60,834 | 39,166 |
| A_balanced | 55,784 | not exported | 0 |
| B_mrmr | 65,447 | not exported | 0 |

## Limitations and handoff

Same hosts/capture may span partitions. No independent PCAP/session IDs. Chronology reduces leakage but is not a held-out network or capture evaluation; attack-family proportions may shift.
All full-file class/quality counts are descriptive auditing, not selection/evaluation evidence. Within-partition multiplicity and conflicting labels are retained; duplicates crossing time partitions are purged from the later partition.
Fit/ranking/model exports use bounded uniform samples. Preprocessing and Track A use only normal training; Track B uses mixed training. Categorical MI considers full nominal codes whereas model encoding caps learned nominal categories; this information loss must be validated.
Normal q0.001/q0.999 are scaling anchors, with training-derived saturation bounds extending the encoded range to [-1,4]. This preserves bounded contrast above the normal upper quantile (e.g. TTL 252 versus 31), although far-tail distinctions still saturate. This v1 policy needs baseline validation. One-hot other is present only if normal frequency is in [0.0005,0.9995], else unknowns map to all zeros. Flag fields expand to all eight specified bits.
Consume `A_compact/preprocessing.json` for exact raw order, formula, encoding, missing behavior and dimension; `baseline.npz` for baseline arrays; `example_raw.csv` + `example_vectors.csv` + `example_labels.csv` for integration; `feature_audit.json` for extraction state/arithmetic/unknown definitions.
Extraction costs are ordinal engineering judgments, separate from software preprocessing and MLP dimension. No LUT/DSP/BRAM/resource estimates or throughput claims without synthesis.
See root `REPORT.md` for interpretation and `README.md` for the single reproduction command and baseline score interface.
