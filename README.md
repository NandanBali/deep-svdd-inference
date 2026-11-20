# NetFlow feature selection for the Pynq-Z2 prototype

The complete trained software baseline is documented in [SOFTWARE.md](SOFTWARE.md).
Run `./reproduce-software.sh --device auto` for training, real evaluation,
verified C++/NumPy exports and the numerical study. The feature-selection workflow
below is the earlier standalone study and preserves its original artifacts.

Run everything from this directory with one command:

```bash
./reproduce.sh
```

This creates `.venv`, installs dependencies pinned in `requirements.lock`, runs synthetic regression tests,
downloads the official **NF-UNSW-NB15-v3** archive if needed, verifies its pinned
BagIt checksums, inspects every CSV row, creates chronological splits, fits the two
selection tracks, freezes a provisional recommendation, and exports integration
artifacts. Network access is needed for installation and the initial download.
The original repository had only the prompt, with no data, loader or evaluator.
v3 was deliberately chosen for its timestamps; the workflow checks the actual
55-column schema and never silently treats original UNSW-NB15 or v2 as v3.

With dependencies already installed:

```bash
python -m svdd_features.workflow --download
```

Parameters are configurable, for example `--seed`, `--sample-rows`, `--mi-rows`,
`--correlation`, `--modal-limit`, `--gap-ms`, `--mrmr-penalty`, `--target-fpr` and
`--output`. Use `--data /path/to/NF-UNSW-NB15-v3.csv` for a local copy instead of
`--download`; its audited schema is required and its SHA-256 is recorded. Unknown
schemas fail explicitly. This implementation has no timestamp-free fallback
because the actual downloaded dataset provides valid timestamps.

The current run uses chunks of 100,000 rows, a uniform sample of at most 100,000
eligible rows per partition, and at most 30,000 mixed training rows for MI. Samples
are without replacement, without class balancing. All normal preprocessing and
Track A fits use the normal rows in the retained training sample. Track B uses
attack training labels explicitly. Test values do not fit or choose anything.
Full-file profile/distribution information is descriptive auditing only.

See [REPORT.md](REPORT.md) for the recommendation and interpretation, and
[results/findings.md](results/findings.md) for reproducibly generated counts.

## Files for the software baseline

Read [results/selection.json](results/selection.json), then each candidate's
`preprocessing.json` and `baseline.npz`. All candidates have the same benign
training row IDs and the same common validation row IDs. Encoded dimensions differ.
Do not independently resplit or refit the exported preprocessing.

```python
from pathlib import Path
import numpy as np
import pandas as pd
from svdd_features import Preprocessor

folder = Path("results/A_compact")
preprocess = Preprocessor.load(folder / "preprocessing.json")
x = preprocess.transform(pd.read_csv(folder / "example_raw.csv"))
gold = pd.read_csv(folder / "example_vectors.csv").to_numpy(np.float32)
np.testing.assert_array_equal(x, gold)

with np.load(folder / "baseline.npz", allow_pickle=False) as data:
    # Copy arrays needed after closing the archive.
    x_train = data["x_train"].copy()        # only normal training samples
    x_validation = data["x_validation"].copy()
    y_validation = data["y_validation"].copy()
```

Every NPZ includes `train_row_ids`, `validation_row_ids`, `attack_validation` and
the arrays above. Only the frozen recommendation includes `x_test`, `y_test`,
`attack_test`, and `test_row_ids`. Row IDs are zero-based parsed CSV row positions,
excluding the header. The full split membership and pre-encoding sample IDs are
saved in `results/splits.npz`; `-1` means purged. These large local arrays and the
dataset are ignored by Git but exist after the command.

Comparison policy: use the same bias-free `d -> 16 -> 8` encoder policy,
activations, initialization policy, training epochs/optimizer/budget, and seeds
(suggested `7,17,27`) for all candidates. Train only `x_train`. Compute a fresh
center for every candidate/seed from its own normal training embeddings, with
the baseline's declared center initialization/clamping policy. Never transfer a
center or a threshold between models. This repository deliberately contains no
trainer inside the feature-selection package. The new `svdd_baseline` package
implements the software experiment in `prompt-software.md`, using its requested
`d -> 32 -> 16 -> 8` architecture; see SOFTWARE.md for that current workflow.

Score convention is squared distance to that model's center; higher is anomalous.
The small integration module supplies normal-validation threshold calibration,
ROC-AUC, trapezoidal PR-AUC, average precision, measured FPR, overall/per-attack
recall, score quantiles, and embedding variance/collapse diagnostics. The initial
target FPR is configurable and defaults to 1%; it is not a project requirement.
Normal validation FPR is a calibration statistic; final test FPR measures held-out
behavior. Recall is undefined if a family has no held-out samples; do not report
missing families as perfect or zero recall.

For each independently trained model save a score NPZ with:

| Key | Value |
|---|---|
| `candidate` | scalar string, e.g. `A_compact` |
| `model_id` | unique scalar string identifying trained weights/run |
| `seed` | scalar integer |
| `contract_sha256` | SHA-256 of that candidate's exact preprocessing JSON bytes |
| `validation_scores` | one score per exported validation row, same order |
| `validation_row_ids` | exact exported row IDs, same order |
| `validation_embeddings`, `center` | recommended: `[N,8]` embeddings and `[8]` center |
| `test_scores`, `test_row_ids` | final frozen model only, after validation selection |

Then evaluate validation scores (without evaluating test):

```bash
python -m svdd_features.evaluation --scores baseline_scores_seed7.npz --freeze results/selection.json --output validation_seed7.json
```

Use validation evidence to choose among the **three existing candidates** and
record any use of attack validation labels. If the preferred choice differs from
this provisional choice, freeze the revised choice and prepare its test export
**before** computing any test scores; do not run a new feature search against test.
The current export only opens test inputs for `A_compact`; extending that export
requires a deliberate baseline handoff change, not relabeling another candidate.

After freezing and training the final model(s), add `--final-test` to that command
once per predeclared seed. A write-once `final_test_seed_<seed>.json` receipt rejects
accidental repeat final evaluation. The workflow refuses to overwrite a directory
with these receipts. Reproduction alone computes **no model test metrics**.

## Files for FPGA integration

- `results/A_compact/preprocessing.json`: ordered raw input contract and saved
  numeric constants. This is the executable artifact; no pickle is needed.
- `results/A_compact/example_raw.csv`, `example_vectors.csv`, `example_labels.csv`:
  normal and per-family attack golden cases from training, with exact float32
  outputs and separate label metadata.
- `results/feature_audit.json`: all 55 columns, sources, units, direction,
  availability, state/update arithmetic, ordinal extraction cost, payload/history
  needs, quality counts, retained candidates and unresolved definitions.
- `sources/provenance.json`, `sources/NetFlow_v3_Features.csv` and
  `sources/manifest-sha1.txt`: exact dataset identity and original definitions.

V1 preprocessing is software: training median imputation, three log1p transforms
for the compact choice, saved training-derived clipping, affine scaling and float32
output. Normal q0.001/q0.999 define scale anchors; saturation bounds extend one
anchor span below and three above that interval, giving numeric output in `[-1,4]`.
This retains bounded contrast outside the normal envelope: TTL 252 remains
different from TTL 31 even when the normal upper quantile is 31. Farther extremes
still saturate. Category/flag outputs are 0/1. These choices require baseline
validation; the contract records every constant.
The encoder still needs its weights, center, activation and numeric precision
contract from the baseline developer. No fixed-point format or FPGA resource
estimate is claimed here. Counter width, flow-table capacity, timestamp precision,
packet loss/expiry behavior and exact nProbe byte/window conventions need a PCAP
equivalence test before replacing software extraction with hardware extraction.

## Verification and outputs

`results/verification.json` records save/reload equality, stable feature ordering,
label exclusion, finite outputs, CSV golden-vector equivalence and cross-partition
encoded-vector separation. It also contains hashes of the exported arrays.
`results/run.json` captures dependency versions, configuration, code hashes and
dataset checksum. `track_a.json`, `A_covariance.csv`, `A_pearson.csv` and
`A_spearman.csv` contain normal-only statistics; `track_b.json` and
`B_mi_ranking.csv` contain the explicitly label-assisted ranking, estimator and
greedy selection steps. `training_diagnostics.png` visualizes training statistics,
not detector performance.

Run `python -m pytest -q` to check synthetic edge cases and a complete offline
synthetic export. Synthetic tests never substitute for real dataset findings.
Source data use UQ's **Permitted Re-Use with Commercial Use Restriction** license;
see the official metadata before redistribution or commercial use.
