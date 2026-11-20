# Deep SVDD software baseline and FPGA handoff

This workflow implements `prompt-software.md`. The original `reproduce.sh`,
`results/` and `REPORT.md` remain the earlier feature-selection study. Software
models and their measured performance are written separately to `runs/software/`.

## Reproduce

From the repository root:

```bash
./reproduce-software.sh --device auto
```

CPU execution, with an independent output directory:

```bash
./reproduce-software.sh --device cpu --output runs/software-cpu
```

The script creates an isolated `.venv-software` with access to compatible local
packages, reuses the installed PyTorch 2.12 wheel when available, installs the
pinned software requirements, runs correctness tests, and executes the complete
experiment. If PyTorch is absent/incompatible, it installs release 2.12 from the
official CPU or CUDA 13.0 wheel index according to local GPU availability. Set
`SVDD_TORCH_INDEX_URL` to choose an official wheel index explicitly. The exact
build, dependencies, GPU and execution device are recorded in `environment.json`.
The system Python environment is never modified. CPU/GPU bitwise training
equivalence is not promised.

With dependencies ready:

```bash
.venv-software/bin/python -m svdd_baseline.run --config configs/software.json --device cuda --output runs/software
```

`--device auto` uses CUDA when available, otherwise CPU. Explicit `cuda` fails
clearly when unavailable. Repeating the same experiment resumes completed,
hash-verified checkpoints and test-score receipts. A changed configuration,
training implementation, model device or frozen identity requires a new output
directory; completed experiments are not silently overwritten.

The dataset defaults to `data/NF-UNSW-NB15-v3.csv`, the previously downloaded and
verified 55-column CSV (53 features plus `Label`/`Attack`). Set `data` in the JSON
configuration for another location. The loader validates the audited schema and
never silently substitutes original UNSW-NB15, v1/v2, or synthetic data. A missing
file is an error. Tests use clearly labeled synthetic fixtures in temporary
directories; those are never reported as dataset performance.

## Protocol and configuration

The workflow reuses the verified dataset fingerprint and chronological split when
the saved split parameters match; otherwise it regenerates that same documented
protocol. The 60/20/20 flow-start cutoffs have a 60-second embargo on either side,
whole-flow end purges and later-partition duplicate-vector purges. Within-partition
multiplicity remains. Related hosts/captures can recur; session isolation cannot
be established from this schema.

Defaults sample 100,000 eligible raw rows per partition, uniformly without
balancing using seed 20261110. Only benign training samples enter preprocessing or
optimization. Validation and test contain both classes. Both configurations use
the same raw evaluation row IDs. Feature-projection/encoding collisions are counted
but are not removed: this software experiment intentionally does not inherit the
previous feature study's additional, feature-dependent validation filtering.

Two configurations are trained:

- **eligible**: every audited eligible feature, including categorical codes,
  directional flags/statistics, application fields and port ranges. Addresses,
  absolute timestamps, transaction identifiers, labels and the two ambiguous
  per-second-byte columns stay excluded.
- **compact**: up to eight low-cost measurements retained by the normal-only
  variance/redundancy/semantic-priority rule. No attack-label MI is used. The actual
  ordered features and encoded indices are saved in `feature_table.json` and each
  preprocessing contract.

The saved preprocessing uses normal medians, audited log1p transforms and
quantile-anchor affine scaling with output saturation in [-1,4]. This departure
from plain standardization is motivated by the observed heavy tails and the loss
of out-of-normal contrast under clipping at the normal upper quantile. For
example, strict clipping would collapse TTL 31 and TTL 252 together. Saturation
still discards distinctions farther into the tail; it is not claimed optimal.

Configuration exposes `scaler` (`quantile_anchor` or `standard`), `log_mode`
(`audited` or `none`), `clip_quantile`, `output_clip` and `clipping`. All fits use
normal training only. Standardization uses fitted mean/population standard
deviation after optional log transforms. Port/protocol semantics and invalid-value
handling remain explicit in the JSON contract.

The encoder is `d -> 32 -> 16 -> 8`, bias-free with two hidden ReLUs and a linear
embedding. The fixed center is initialized before optimization from mean normal
embeddings, with signed epsilon treatment (epsilon .1; exact zero becomes +.1).
Adam optimizes mean squared distance plus `lambda/2 * sum(W^2)`, lambda 1e-6, with
optimizer weight decay zero. No AMP, TF32, normalization, dropout, reconstruction
decoder, pretraining or soft-boundary objective is used.

Each model runs at most 30 epochs, with minimum five and patience five on normal
validation distance. The literal lowest-distance checkpoint is restored.
Parameters, center and threshold are specific to every seed/model. Defaults run
eligible seed 7 and compact seeds 7,17,27. Compact seed 7 is the predeclared primary
export; it is not chosen using the best test result. No attack validation label
influences feature, architecture, checkpoint or threshold selection.

Normal validation calibrates the threshold at target FPR .01 using quantile method
`higher` and strict `score > threshold`. Ties are conservative. Before loading test
raw features, `freeze.json` records all four checkpoints, contracts, centers,
thresholds, test row-ID digest and the predeclared numerical-study formats and
normal-validation quantized thresholds. Test results cannot change those choices.

## Results and checkpoints

Start with `runs/software/report.md` and `summary.json`. Other important files:

| File | Purpose |
|---|---|
| `dataset_audit.json`, `split_manifest.json`, `data/split_indices.npz` | Exact source identity, full audit, split policy and zero-based CSV row IDs |
| `feature_table.json` / `.csv` | Inclusion/exclusion, meaning, encoding, model index, extraction concerns |
| `data/*_preprocessing.json` | Complete normal-fitted preprocessing contracts |
| `<configuration>_seed<seed>/model.pt` | PyTorch state_dict, fixed center, seed and experiment identity |
| `training.json`, `history.json` | Loss/optimizer/stopping policy, actual epochs/duration, collapse diagnostics and learning curves |
| `threshold.json`, `validation_scores.npz` | Per-model normal calibration and complete validation traces |
| `evaluation.json`, `test_scores.npz`, `test_receipt.json` | Held-out metrics, traces/labels/row IDs and frozen-model receipt |
| `benchmark.json`, `preprocessing_timings.json` | Separate network, preprocessing and CSV loading measurements |
| `training_and_scores.png` | Real learning curves and held-out normal/attack score distributions |
| `numerical_study.json` | Ranges/headroom, accumulator widths, quantized errors/saturation and metrics |
| `diagnostics.md`, `diagnostics.json` | Post-hoc overfitting, contraction, shortcut, host-overlap and novel-input audits |
| `environment.json`, `completion.json` | Package/device/source versions, hashes and completion status |

Metrics include AUROC, explicitly named average precision, separately named
trapezoidal PR-AUC, precision/recall/F1/FPR, confusion counts, attack prevalence and
per-family recall. Seed summaries report population standard deviation, not a
confidence interval. Tiny family counts and embedding-collapse warnings remain
visible. A complete run does not imply good anomaly-detection performance.

Network CPU timings use one CPU thread, inference mode and preloaded tensors.
Five trials measure batch-one and batch-1024 network-plus-distance calls after
warm-up. Reported p95 is the p95 of trial means, not individual-call tail latency.
Preprocessing and data loading have separate measurements. No FPGA throughput is
inferred from these timings.

## FPGA developer: use the primary compact export

Use `runs/software/compact_seed7/export/` and its paired parent checkpoint:

- `metadata.json`: ordered PL inputs, model/checkpoint identity, dimensions,
  float32 types, center/threshold, operations and hashes.
- `preprocessing.json`: PS-side raw feature ordering, transforms, constants,
  missing/invalid behavior and output ordering.
- `parameters.npz` and `W0.npy`, `W1.npy`, `W2.npy`, `center.npy`, `threshold.npy`:
  independent NumPy-readable arrays; no PyTorch is needed to inspect them.
- `model_parameters.h`: generated float32 C++ weights, center and threshold.
  Matrices are row-major `W[out][in]`, and inference is `y = W*x`.
- `reference.cpp` / `reference`: portable C++17 reference and compiled executable.
- `golden_raw.csv`, `golden_inputs.csv`, `golden_labels.csv`,
  `golden_expected.npz`, `golden_outputs.csv`, `golden_cpp_outputs.csv`:
  256 final-test integration cases, including normal/attack coverage, every
  available family, and 32 closest cases on each threshold side when available.
- `reference_verification.json`: independent NumPy/C++ error maxima and every
  threshold-borderline disagreement. Nonborderline decision differences fail.

`golden_raw.csv` has column headers. Network input/output numeric CSVs are
headerless. Inputs follow `ordered_pl_inputs` in metadata. Output columns are:
32 first hidden ReLU values, 16 second hidden ReLU values, eight embeddings,
eight squared distance terms, one score, then one decision (66 columns total).
The NPZ uses named arrays and is the easiest integration-test source.

The PL contract is:

```text
preprocessed float32 vector
  -> dense 32 -> ReLU -> dense 16 -> ReLU -> dense 8
  -> sum((embedding - fixed_center)^2)
  -> anomaly iff score > frozen_threshold
```

Initially preprocessing runs on the PS. Packet parsing, feature extraction,
training and online updates are outside this PL core. No Vivado/Vitis installation
is needed for the software handoff.

Compile/recheck the portable reference from its export directory:

```bash
g++ -std=c++17 -O2 -ffp-contract=off -fno-fast-math reference.cpp -o reference
./reference golden_inputs.csv > checked_outputs.csv
```

The independent Python reference is
`svdd_baseline.reference.numpy_infer(parameters, inputs)`; use `load_parameters`
to validate hashes, dimensions and matrix orientation first. Activation and score
tolerances are recorded separately. Near-center distance terms have a separate
absolute tolerance because squaring a cancellation amplifies relative error.
Every borderline decision mismatch is disclosed; agreement is not called bitwise
equivalence.

The workflow also audits the frozen models after evaluation. Supervised
single-feature probes use mixed training labels and validation only; validation
permutations test sensitivity without refitting the neural model. Separate test
strata and nominal binomial intervals are exploratory diagnostics. These checks
never alter weights, preprocessing or thresholds, and do not create a new
independent final test. Re-run just the audit with:

```bash
.venv-software/bin/python -m svdd_baseline.diagnostics --output runs/software
```

## Fixed-point study

Use `runs/software/compact_seed7/export/fixed_point/Q24_I8/model_parameters.h`
for the 24-bit HLS constants. The same directory contains the format metadata and
256 integer golden inputs and outputs. The sibling `Q16_I6` directory contains
the 16-bit variant. Regenerate both from the frozen run with
`.venv-software/bin/python -m svdd_baseline.export_fixed`.

The compact core consumes eight preprocessed features in `ordered_pl_inputs`
order. It applies bias-free dense layers of 32, 16 and 8 units, with ReLU after
the first two layers, then returns the sum of squared differences between the
eight embedding coordinates and the fixed center. The matrices have shapes
`[32][8]`, `[16][32]` and `[8][16]` in `W[out][in]` order, for 896 weights.

- `W0`, `W1`, `W2` and `CENTER` hold signed raw codes with `FRACTIONAL_BITS`
  fractional bits. A code `q` represents `q / 2^F`. When using `ap_fixed`,
  assign the raw bits into the fixed-point word to preserve that scale.
- Accumulate each complete dot product at `2F` fractional bits using the
  corresponding `W0_ACC_BITS`, `W1_ACC_BITS` or `W2_ACC_BITS` width. Cast
  once to the layer format with nearest-even rounding and saturation
  (`AP_RND_CONV`, `AP_SAT`), then apply the hidden ReLU.
- Widen the center subtraction by one bit. Sum the eight squares at `2F`
  fractional bits using `SCORE_ACC_BITS`. Flag an anomaly on
  `score_code > THRESHOLD`. The header also includes `VALIDATION_THRESHOLD`
  for the separately calibrated operating point. For Q24_I8 the two threshold
  codes are 6000531 and 6018545, respectively.
- Both golden CSVs are headerless. Each output row has 32 first hidden codes,
  16 second hidden codes, eight embedding codes, eight square codes, one score
  code, and two decision bits in original-threshold and validation-threshold
  order. Input, hidden and embedding codes use scale `2^F`. Square and score
  codes use scale `2^(2F)`.

The study tests predeclared signed 16/6 and 24/8 total/integer-bit formats (integer
bits include sign), nearest-even rounding, saturating input/weight/center and layer
output casts, and widened exact integer products/accumulation. Hidden ReLU follows
the cast. Center subtraction widens by one bit; distance terms and the score retain
2F fractional bits in int64 rather than being narrowed to the activation format.

Original-threshold metrics are evaluated first without threshold rounding. A
second threshold is calibrated from normal **validation** quantized scores and
reported separately. Suggested formats from measured ranges include 25% headroom;
they do not alter the two predeclared test formats. This experiment is not
bit-accurate Vitis `ap_fixed` verification, synthesis, or a resource estimate.

Before committing a fixed-point FPGA format, reproduce casts/rounding/saturation
in Vitis 2023.2, check the golden vectors, synthesize and measure on Pynq-Z2.
