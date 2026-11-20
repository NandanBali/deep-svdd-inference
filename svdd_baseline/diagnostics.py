"""Post-hoc audit of frozen models. Never trains or retunes the baseline."""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import ks_2samp, beta
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

from svdd_features.data import load_ids, disjoint_encoded
from svdd_features.preprocessing import Preprocessor
from .common import save_json, sha
from .evaluation import evaluate
from .training import load_model, trace_numpy, configure


def interval(success, total):
    if not total:
        return [None, None]
    return [0. if success == 0 else float(beta.ppf(.025, success, total-success+1)),
            1. if success == total else float(beta.ppf(.975, success+1, total-success))]


def describe(scores):
    return dict(rows=len(scores), mean=float(np.mean(scores)),
        quantiles=dict(zip(["min", "median", "q95", "q99", "q999", "max"],
                           np.quantile(scores, [0, .5, .95, .99, .999, 1]).tolist())))


def cohort(scores, labels, attacks, threshold):
    if not len(scores):
        return {"rows": 0}
    if len(np.unique(labels)) == 2:
        return evaluate(scores, labels, attacks, threshold)
    normal, attack = labels == 0, labels == 1
    pred = scores > threshold
    return dict(rows=len(labels), normal_rows=int(normal.sum()), attack_rows=int(attack.sum()),
        roc_auc=None, average_precision=None,
        measured_fpr=float(pred[normal].mean()) if normal.any() else None,
        recall=float(pred[attack].mean()) if attack.any() else None,
        reason="Ranking metrics undefined in a single-class cohort")


def audit(output):
    out = Path(output)
    freeze_bytes = (out/"freeze.json").read_bytes()
    freeze = json.loads(freeze_bytes)
    config = json.loads((out/"config.json").read_text())
    with np.load(out/"data/split_indices.npz", allow_pickle=False) as archive:
        ids = {k: archive[k+"_row_ids"] for k in ["train", "validation", "test"]}
    frames = load_ids(Path(config["data"]), ids, config["chunk_size"])
    primary_name = freeze["primary_model"]
    primary_folder = out/primary_name
    pipe = Preprocessor.load(primary_folder/"export/preprocessing.json")
    train, validation, test = [frames[k] for k in ["train", "validation", "test"]]
    xtr, xv, xt = [pipe.transform(f) for f in [train, validation, test]]
    ytr, yv, yt = [f.Label.to_numpy(np.uint8) for f in [train, validation, test]]
    av, at = validation.Attack.to_numpy(dtype=str), test.Attack.to_numpy(dtype=str)
    recorded_device = json.loads((primary_folder/"training.json").read_text())["device"]
    diagnostic_device = recorded_device if recorded_device != "cuda" or torch.cuda.is_available() else "cpu"
    model, metadata = load_model(primary_folder, diagnostic_device)
    device = str(next(model.parameters()).device)
    configure(config["primary_seed"], config["training"]["cpu_threads"])
    train_scores = trace_numpy(model, xtr, device)["scores"]
    with np.load(primary_folder/"validation_scores.npz", allow_pickle=False) as z:
        np.testing.assert_array_equal(z["row_ids"], validation.index.to_numpy())
        val_scores = z["scores"]
    with np.load(primary_folder/"test_scores.npz", allow_pickle=False) as z:
        np.testing.assert_array_equal(z["row_ids"], test.index.to_numpy())
        test_scores = z["scores"]
    threshold = freeze["models"][primary_name]["threshold"]
    result = dict(experiment_id=freeze["experiment_id"], freeze_sha256=sha(out/"freeze.json"),
        primary_model=primary_name, checkpoint_sha256=metadata["checkpoint_sha256"], diagnostic_device=device,
        scope="Post-hoc diagnostics only; baseline weights, preprocessing, thresholds and saved predictions unchanged. Supervised probes use training labels and validation scores only, never test scores.",
        normal_score_generalization={}, learning_curves={}, embedding_contraction={},
        label_assisted_validation_probes=[], validation_permutation_sensitivity=[], host_overlap={}, ttl_tables={})
    for name, scores, y in [("train", train_scores, ytr), ("validation", val_scores, yv), ("test", test_scores, yt)]:
        normal = scores[y == 0]
        result["normal_score_generalization"][name] = describe(normal)
        result["normal_score_generalization"][name]["fpr_at_frozen_threshold"] = float(np.mean(normal > threshold))
    for name, scores, y in [("validation", val_scores, yv), ("test", test_scores, yt)]:
        result["normal_score_generalization"][name]["ks_distance_from_train"] = float(ks_2samp(train_scores[ytr == 0], scores[y == 0], method="asymp").statistic)
    for name in freeze["models"]:
        training = json.loads((out/name/"training.json").read_text())
        history = json.loads((out/name/"history.json").read_text())
        chosen = history[training["best_epoch"]-1]
        result["learning_curves"][name] = dict(best_epoch=training["best_epoch"], epochs_run=training["epochs_run"],
            best_at_last_epoch=training["best_epoch"] == training["epochs_run"],
            normal_validation_to_training_loss_ratio=chosen["normal_validation_distance"]/max(chosen["training_distance"], 1e-30),
            note="Training loss is averaged while weights change; ratio is descriptive, not a paired fixed-checkpoint estimate")
        d = training["embedding_diagnostics"]
        result["embedding_contraction"][name] = dict(final_total_variance=d["total_variance"],
            final_to_initial_variance_ratio=d["variance_ratio_to_initial"],
            contraction_factor=1/max(d["variance_ratio_to_initial"], 1e-30),
            absolute_collapse_warning=d["possible_collapse"], relative_collapse_warning=d["relative_collapse_warning"],
            interpretation="Strong contraction is expected under SVDD's objective but is also a collapse warning sign. Passing a fixed heuristic is not proof of useful representation.")
    # Probes are explicitly supervised, strictly post-hoc, and validation-only.
    for i, feature in enumerate(pipe.contract["output_names"]):
        if len(np.unique(ytr)) == 2:
            stump = DecisionTreeClassifier(max_depth=1, random_state=config["data_seed"])
            stump.fit(xtr[:, [i]], ytr)
            probability = stump.predict_proba(xv[:, [i]])[:, list(stump.classes_).index(1)]
            result["label_assisted_validation_probes"].append(dict(feature=feature,
                validation_roc_auc=float(roc_auc_score(yv, probability)),
                validation_average_precision=float(average_precision_score(yv, probability)),
                learned_split_in_preprocessed_units=float(stump.tree_.threshold[0]),
                supervised_training_rows=len(ytr), baseline_training_changed=False, test_evaluated=False))
        shuffled = xv.copy()
        shuffled[:, i] = np.random.default_rng(config["data_seed"]+i).permutation(shuffled[:, i])
        permuted = trace_numpy(model, shuffled, device)["scores"]
        result["validation_permutation_sensitivity"].append(dict(feature=feature,
            validation_roc_auc=float(roc_auc_score(yv, permuted)),
            validation_average_precision=float(average_precision_score(yv, permuted)),
            caution="Marginal permutation breaks feature dependencies; sensitivity is not causal importance"))
    result["label_assisted_validation_probes"].sort(key=lambda r: -r["validation_roc_auc"])
    result["baseline_validation_roc_auc"] = float(roc_auc_score(yv, val_scores))
    result["baseline_validation_average_precision"] = float(average_precision_score(yv, val_scores))
    for column in ["IPV4_SRC_ADDR", "IPV4_DST_ADDR"]:
        known = set(train[column])
        result["host_overlap"][column] = dict(training_unique=len(known),
            validation_row_fraction_seen_in_training=float(validation[column].isin(known).mean()),
            test_row_fraction_seen_in_training=float(test[column].isin(known).mean()),
            input_to_model=False)
    result["host_overlap"]["interpretation"] = "Excluded IPs do not establish topology independence; TTL/window/size features can remain host or scenario proxies. An independent-capture/network evaluation is still needed."
    for name, f in frames.items():
        table = pd.crosstab(f.MIN_TTL, f.Label)
        result["ttl_tables"][name] = [{"ttl": float(value), "normal": int(row.get(0., 0)), "attack": int(row.get(1., 0))} for value, row in table.iterrows()]
    normal_x = xtr[ytr == 0]
    novel = disjoint_encoded([normal_x], xt)
    result["test_projected_input_strata"] = dict(
        definition="Exact float32 preprocessed-vector match to normal training, not exact raw-flow duplication; post-hoc exploratory test strata",
        repeated=cohort(test_scores[~novel], yt[~novel], at[~novel], threshold),
        novel=cohort(test_scores[novel], yt[novel], at[novel], threshold))
    pred = test_scores > threshold
    result["attack_recall_uncertainty"] = {}
    for family in np.unique(at[yt == 1]):
        mask = (at == family) & (yt == 1)
        hits, n = int(pred[mask].sum()), int(mask.sum())
        result["attack_recall_uncertainty"][str(family)] = dict(rows=n, detected=hits, recall=hits/n,
            nominal_binomial_95_interval=interval(hits, n))
    normal = yt == 0
    result["test_fpr_nominal_binomial_95_interval"] = interval(int(pred[normal].sum()), int(normal.sum()))
    result["uncertainty_caution"] = "Clopper-Pearson intervals assume independent Bernoulli trials. Related/duplicate flows violate that assumption; these nominal intervals can understate deployment uncertainty. Seed variation and dataset/capture shift are separate concerns."
    assert (out/"freeze.json").read_bytes() == freeze_bytes
    result["diagnostic_source_sha256"] = sha(Path(__file__))
    save_json(out/"diagnostics.json", result)
    write_diagnostics(out, result)
    print(json.dumps({k: result[k] for k in ["normal_score_generalization", "label_assisted_validation_probes", "test_projected_input_strata"]}, indent=2), flush=True)
    return result


def write_diagnostics(out, r):
    lines = ["# Frozen-model overfitting and shortcut audit", "", r["scope"], "",
        "## Generalization", "", "| Normal cohort | Rows | Mean score | 99th percentile | FPR at frozen threshold | KS distance from training |", "|---|---:|---:|---:|---:|---:|"]
    for name, x in r["normal_score_generalization"].items():
        lines.append(f"| {name} | {x['rows']} | {x['mean']:.7g} | {x['quantiles']['q99']:.7g} | {x['fpr_at_frozen_threshold']:.4%} | {x.get('ks_distance_from_train', 0):.4f} |")
    lines += ["", "Validation and test gaps are descriptive checks, not permission to retune using test. Low normal validation loss alone does not establish attack generalization.", "",
        "## Representation contraction", "", "| Model | Final / initial variance | Contraction factor | Absolute / relative warning |", "|---|---:|---:|---|"]
    for name, d in r["embedding_contraction"].items():
        lines.append(f"| {name} | {d['final_to_initial_variance_ratio']:.7g} | {d['contraction_factor']:.1f}x | {d['absolute_collapse_warning']} / {d['relative_collapse_warning']} |")
    lines += ["", "All embeddings contract substantially. The saved fixed cutoffs are warning heuristics, not proof that collapse is absent. Inspect contraction alongside score separation, seed variation and the training curves.", "",
        "## Label-assisted validation-only shortcut probes", "",
        "One supervised depth-one tree per preprocessed input is fitted to mixed training labels and evaluated on validation only. These trees are post-hoc diagnostics, not the normal-only Deep SVDD model or a new feature selector; none is evaluated on test.", "",
        "| Input | Validation AUROC | Validation AP | Split in preprocessed units |", "|---|---:|---:|---:|"]
    for p in r["label_assisted_validation_probes"]:
        lines.append(f"| {p['feature']} | {p['validation_roc_auc']:.6f} | {p['validation_average_precision']:.6f} | {p['learned_split_in_preprocessed_units']:.6g} |")
    lines += ["", f"Baseline validation AUROC/AP: {r['baseline_validation_roc_auc']:.6f} / {r['baseline_validation_average_precision']:.6f}.",
        "Permutation sensitivities and raw class-by-TTL tables are in diagnostics.json. Permutation breaks dependencies and should not be read as causal importance.", "",
        "## Host and projected-input overlap", ""]
    for col, d in r["host_overlap"].items():
        if isinstance(d, dict):
            lines.append(f"- {col}: {d['training_unique']} distinct training values; {d['validation_row_fraction_seen_in_training']:.2%} validation rows and {d['test_row_fraction_seen_in_training']:.2%} test rows have a value seen in training. These fields never enter the model.")
    lines += ["", r["host_overlap"]["interpretation"], "",
        "Post-hoc test strata use the original frozen threshold and unchanged saved scores:", "",
        "| Test encoded vectors | Rows | AUROC | AP | Recall | FPR |", "|---|---:|---:|---:|---:|---:|"]
    for name, d in r["test_projected_input_strata"].items():
        if isinstance(d, dict):
            def fmt(key):
                value = d.get(key)
                return "undefined" if value is None else f"{value:.6f}"
            lines.append(f"| {name} | {d['rows']} | {fmt('roc_auc')} | {fmt('average_precision')} | {fmt('recall')} | {fmt('measured_fpr')} |")
    lines += ["", "An identical projected vector is not necessarily an identical raw flow. These strata expose repeated benign patterns and conditional FPR; they do not replace the original test result or justify claims of independent deployment performance.", "",
        "## Rare-attack uncertainty", "", "| Attack | Examples | Recall | Nominal 95% binomial interval |", "|---|---:|---:|---|"]
    for name, d in r["attack_recall_uncertainty"].items():
        lo, hi = d["nominal_binomial_95_interval"]
        lines.append(f"| {name} | {d['rows']} | {d['recall']:.4%} | [{lo:.4%}, {hi:.4%}] |")
    lines += ["", r["uncertainty_caution"], "",
        "## Interpretation and next experiment", "",
        "The normal-training versus normal-validation/test distributions do not by themselves show a classic optimistic training-loss gap. The stronger concerns are shortcut separability, almost complete host overlap, substantial representation contraction and higher conditional FPR on novel inputs. None of these checks establishes independent-network generalization.",
        "High accuracy on these chronological cohorts is not sufficient to rule out dataset shortcuts, capture dependence or representation contraction. This audit changes no trained model, threshold or primary export.",
        "The next independent experiment should hold out a separate capture/network, assess TTL/topology reliance using validation-only ablations, and predeclare calibration/evaluation treatment of novel flow patterns. Any architecture/pretraining changes need fresh development validation and a new independent final test; the current test is now consumed.", ""]
    (out/"diagnostics.md").write_text("\n".join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("runs/software"))
    args = p.parse_args()
    audit(args.output)


if __name__ == "__main__":
    main()
