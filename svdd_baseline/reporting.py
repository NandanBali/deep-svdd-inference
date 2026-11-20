from __future__ import annotations

import json
from pathlib import Path
import numpy as np


def plots(out, records):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(records), 2, figsize=(12, 3.5*len(records)), layout="constrained", squeeze=False)
    for row, (name, record) in enumerate(records.items()):
        folder = out/name
        h = json.loads((folder/"history.json").read_text())
        ax = axes[row, 0]
        ax.plot([v["epoch"] for v in h], [v["training_distance"] for v in h], label="normal training distance")
        ax.plot([v["epoch"] for v in h], [v["normal_validation_distance"] for v in h], label="normal validation distance")
        ax.set(xlabel="Epoch", ylabel="Mean squared distance", title=name)
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        with np.load(folder/"test_scores.npz", allow_pickle=False) as z:
            scores, y = z["scores"], z["labels"]
        ax = axes[row, 1]
        bins = np.linspace(np.log10(max(float(scores.min()), 1e-15)), np.log10(max(float(scores.max()), 1e-14)), 70)
        for k, label, color in [(0, "normal", "#326e8d"), (1, "attack", "#c55a35")]:
            ax.hist(np.log10(np.maximum(scores[y == k], 1e-15)), bins=bins, density=True, alpha=.55, label=label, color=color)
        ax.axvline(np.log10(max(record["calibration"]["threshold"], 1e-15)), color="black", linestyle="--", label="frozen threshold")
        ax.set(xlabel="log10(score), floor 1e-15 for display", ylabel="Density", title="Held-out scores (not used for tuning)")
        ax.legend(fontsize=8)
    fig.savefig(out/"training_and_scores.png", dpi=150)
    plt.close(fig)


def write_report(out, summary):
    config, dataset, split = [summary[k] for k in ["config", "dataset", "split"]]
    lines = ["# Deep SVDD software baseline", "",
        "**SYNTHETIC SMOKE TEST ONLY**" if config["synthetic"] else "**Real NF-UNSW-NB15-v3 experiment.**",
        "", f"Dataset: `{config['data']}`; SHA-256 `{dataset['csv_sha256']}`. Full audit: {dataset['rows']:,} rows, 53 raw features plus binary/family labels; classes {dataset['class_distribution']}.",
        "The verified CSV labels are preserved. The published website's family-count discrepancy is documented in the original feature report. Full feature types, missing/infinite values, constants and duplicates are in dataset_audit.json.", "",
        "## Protocol", "",
        "Reused chronological 60/20/20 split, 60-second symmetric embargo and whole-flow boundary purge. Later repetitions of eligible raw feature vectors are purged from later partitions; within-partition multiplicity remains. Same hosts/capture can recur; no claim of complete session isolation.",
        f"Unbalanced bounded sampling, seed {config['data_seed']}: {split['software_sample']['sizes']}; normal training rows {split['software_sample']['normal_training_rows']:,}. Both configurations use identical validation/test raw row IDs. Training attacks do not enter optimization or preprocessing fits.",
        "This software evaluation retains projected-input collisions introduced by feature reduction/encoding. It does not reuse the previous feature study's additionally purged 33,310-row validation cohort. Results remain conditional on the raw split purge and bounded sample; they are not an independent-network deployment estimate.",
        "No attack labels select features, architecture, epoch, threshold or primary seed. Epoch selection uses normal validation distance only. All four float models and both predeclared quantization formats/calibrations were frozen before test scoring.", "",
        "## Preprocessing and architecture", "",
        "All preprocessing fits normal training only. Defaults reuse audited log transforms and normal q0.001/q0.999 anchors because the data contain heavy-tailed counters/rates and a hard normal-bound clip would discard useful out-of-normal contrast. Numeric saturation is [-1,4]; one-hot codes and flag bits are 0/1. Configuration also supports standardization, disabling logs and changing/disabling clipping. Required missing columns fail; invalid values use the saved median/category/mask rules.",
        "Raw IPs, timestamps, labels and transaction identifiers are excluded. Two ambiguous per-second-byte fields are excluded for definition and data-quality problems. Ports use explicit ranges; protocol/application codes are nominal. The feature table records every decision, encoding, index and extraction concern.",
        "Architecture: d -> 32 -> 16 -> 8; all layers bias-free; two hidden ReLUs, linear embedding; no normalization, dropout or decoder. Float32, no AMP/TF32. Hidden weights use Kaiming uniform; final weights use Xavier uniform.",
        "Fixed center: mean initial normal-training embeddings, near-zero components clamped to signed epsilon=0.1 (exact zero -> +0.1). The center never updates. Objective is mean squared distance + lambda/2 * sum(weight^2), lambda=1e-6; Adam learning rate 1e-3, betas (0.9,0.999), weight_decay=0. No duplicate L2 term. All resolved settings are in each training.json.",
        "Up to 30 epochs, minimum 5, patience 5 for relative normal-validation improvement >=1e-4; restore literal lowest normal-validation-distance checkpoint. Collapse/variance diagnostics and hidden zero fractions remain visible even if metrics are poor.",
        f"Target FPR {config['target_fpr']:.2%}; threshold uses normal-validation quantile with method='higher', anomaly iff score > threshold. Ties are conservative. Each model keeps its own center/threshold. The compact seed-{config['primary_seed']} model was predeclared for FPGA handoff, not chosen using test or best-seed performance.", "",
        "## Measured test performance", "",
        "| Run | Inputs | AUROC | Average precision | Precision | Recall | F1 | FPR | Threshold | Train seconds | Parameters / MACs |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, r in summary["models"].items():
        m, t = r["test"], r["training"]
        lines.append(f"| {name} | {t['input_dimension']} | {m['roc_auc']:.6f} | {m['average_precision']:.6f} | {m['precision']:.6f} | {m['recall']:.6f} | {m['f1']:.6f} | {m['measured_fpr']:.6f} | {r['calibration']['threshold']:.9g} | {t['training_seconds']:.3f} | {t['parameter_count']} / {t['dense_macs_per_flow']} |")
    lines += ["", "Average precision is reported as average precision; trapezoidal PR-AUC is separately named in JSON. MAC counts cover dense layers only, excluding ReLU and eight distance squares/additions. Software timings do not predict FPGA throughput.", ""]
    for name, r in summary["models"].items():
        m, b, d = r["test"], r["benchmark"], r["training"]["embedding_diagnostics"]
        lines += [f"### {name}", "", f"Test rows {m['rows']:,}; attack prevalence {m['attack_prevalence']:.4%}; confusion matrix [true benign, true attack] by [predicted benign, predicted attack]: `{m['confusion_matrix']['matrix']}`.",
            f"Normal calibration rows {r['calibration']['normal_calibration_rows']:,}; achieved normal-validation FPR {r['calibration']['measured_calibration_fpr']:.6f}. Best epoch {r['training']['best_epoch']} of {r['training']['epochs_run']}.",
            f"Embedding total variance {d['total_variance']:.6g}, ratio to initial {d['variance_ratio_to_initial']:.6g}; absolute collapse warning {d['possible_collapse']}, relative collapse warning {d['relative_collapse_warning']}. These scale-dependent heuristics do not prove absence of collapse.",
            f"CPU network+score: batch-one median {b['batch_one']['median_call_ns']/1000:.3f} us; batch-{b['batched']['batch_size']} median {b['batched']['median_ns_per_flow']/1000:.3f} us/flow. {b['methodology']}", "",
            "| Attack | Test examples | Recall |", "|---|---:|---:|"]
        lines += [f"| {a} | {v['rows']} | {v['recall']:.6f} |" for a, v in m["per_attack"].items()]
        lines += [""]
    lines += ["## Compact-seed variation", "", "Population standard deviation across the three predeclared seeds (not a confidence interval):", "", "| Metric | Mean | Stddev |", "|---|---:|---:|"]
    lines += [f"| {k} | {v['mean']:.6f} | {v['std']:.6f} |" for k, v in summary["compact_seed_summary"].items()]
    lines += ["", "## Numerical study", "",
        "Signed 16/6 and 24/8 total/integer-bit formats include the sign bit. Round nearest-even; saturate inputs, weights, center and layer output casts; use exact widened products/sums in the int64 experiment. ReLU follows hidden output casts. Center subtraction widens by one bit; distance products and sum retain 2F fractional bits. Per-layer conservative accumulator widths, observed ranges, 25% range headroom suggestions and saturation counts are in numerical_study.json.",
        "This is an explicit arithmetic simulation, **not bit-accurate HLS verification**. Test metrics use the original threshold first, then the separately frozen normal-validation quantized calibration. Formats were not tuned on test results.", "",
        "| Format | Score MAE | Max score error | Original-threshold disagreements | Saturated data values | AUROC | AP | Recall / FPR (original) | Recall / FPR (recalibrated) |", "|---|---:|---:|---:|---:|---:|---:|---|---|"]
    for name, q in summary["quantized"].items():
        m, c = q["original_threshold_metrics"], q["recalibrated_metrics"]
        lines.append(f"| {name} | {q['score_mae']:.6g} | {q['max_absolute_score_error']:.6g} | {q['decision_disagreements']} | {q['data_saturation_count']} | {m['roc_auc']:.6f} | {m['average_precision']:.6f} | {m['recall']:.6f} / {m['measured_fpr']:.6f} | {c['recall']:.6f} / {c['measured_fpr']:.6f} |")
    lines += ["", "## FPGA handoff and verification", "",
        f"Primary export: `compact_seed{config['primary_seed']}/export/`. It contains `model_parameters.h`, `parameters.npz`, individual `.npy` arrays, `preprocessing.json`, `metadata.json`, a portable `reference.cpp`, raw/normalized golden cases and expected hidden activations, embeddings, distance terms, scores and decisions. The paired PyTorch checkpoint is in its parent model directory.",
        "PL receives the already-preprocessed float32 vector; PS may execute saved preprocessing. Matrices are W[out,in], row-major. Compute dense/ReLU/dense/ReLU/dense, squared distance to the fixed center, then strict threshold comparison. No packet parser, training or online update is implied.",
        "Independent NumPy and compiled C++ traces are checked against PyTorch at every layer. Verification reports contain error maxima, exact compiler flags, model/checkpoint identities and all threshold-borderline decision disagreements. No borderline case is silently removed. Nonborderline decision differences fail the workflow.",
        "CPU and CUDA model execution, center immutability, export identities, data ordering, calibration without test labels, quantization arithmetic and full synthetic workflow behavior are covered by focused tests. Cross-device bitwise training equivalence is not claimed. Training seconds cover optimization epochs and their validation checks, excluding preprocessing, center initialization and export.",
        "Overfitting/shortcut audit: see diagnostics.md and diagnostics.json for normal train/validation/test score gaps, representation contraction, post-hoc validation-only supervised probes and permutation sensitivity, host overlap, novel-versus-repeated input strata and nominal rare-family intervals. These diagnostics never update the baseline. High benchmark scores alone do not establish cross-network generalization.",
        "Reproduce: `./reproduce-software.sh --device auto`; CPU: `./reproduce-software.sh --device cpu --output runs/software-cpu`. Resume an interrupted matching run with the same command; completed checkpoints/test receipts are verified and reused. A changed experiment requires another output directory.",
        "Full results: summary.json, dataset_audit.json, split_manifest.json, feature_table.json, freeze.json, numerical_study.json, preprocessing_timings.json and environment.json. Each model directory contains training.json, history.json, model.pt, threshold.json, evaluation.json, scores, benchmark.json and the export.",
        "Remaining hardware work: choose and validate Vitis 2023.2 numeric formats and interface/control design, synthesize and verify on Pynq-Z2. Floating-point reference correctness is not proof of fixed-point HLS equivalence or a resource/throughput guarantee.", ""]
    (out/"report.md").write_text("\n".join(lines))
