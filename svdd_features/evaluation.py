"""Small score/embedding integration API for the independently developed baseline.

Higher scores mean more anomalous. Model-specific centers and thresholds belong
to the baseline's run, never to another candidate or seed. No model is trained here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, auc, roc_auc_score


def _scores(scores):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Expected nonempty finite one-dimensional scores")
    return scores


def calibrate(normal_validation_scores, target_fpr=.01):
    x = _scores(normal_validation_scores)
    if not 0 < target_fpr < 1:
        raise ValueError("Target FPR must be between zero and one")
    # The strict > comparison below makes tied quantiles conservative.
    threshold = float(np.quantile(x, 1-target_fpr, method="higher"))
    return dict(threshold=threshold, target_fpr=target_fpr, normal_calibration_rows=len(x),
                measured_calibration_fpr=float(np.mean(x > threshold)), comparison="score > threshold")


def metrics(scores, labels, attacks, threshold):
    x = _scores(scores)
    labels, attacks = np.asarray(labels), np.asarray(attacks)
    if labels.ndim != 1 or attacks.ndim != 1 or len(x) != len(labels) or len(x) != len(attacks):
        raise ValueError("Scores, labels and attack names must align")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Both normal and attack labels are required for ROC/PR")
    pred = x > threshold
    precision, recall, _ = precision_recall_curve(labels, x)
    def distribution(mask):
        return dict(zip(["min", "q01", "median", "q99", "max"], np.quantile(x[mask], [0, .01, .5, .99, 1]).tolist()))
    return dict(roc_auc=float(roc_auc_score(labels, x)),
        pr_auc_trapezoidal=float(auc(recall, precision)), average_precision=float(average_precision_score(labels, x)),
        threshold=float(threshold), recall=float(pred[labels == 1].mean()), measured_fpr=float(pred[labels == 0].mean()),
        per_attack={str(a): dict(rows=int(((attacks == a) & (labels == 1)).sum()),
            recall=float(pred[(attacks == a) & (labels == 1)].mean())) for a in np.unique(attacks[labels == 1])},
        normal_score_distribution=distribution(labels == 0), attack_score_distribution=distribution(labels == 1),
        score_std=float(x.std()), nearly_constant_scores=bool(x.std() < 1e-10))


def embedding_diagnostics(embeddings, center):
    z, c = np.asarray(embeddings, dtype=float), np.asarray(center, dtype=float)
    if z.ndim != 2 or c.shape != (z.shape[1],) or not len(z) or not np.isfinite(z).all() or not np.isfinite(c).all():
        raise ValueError("Need finite [n,embedding_dim] embeddings and matching model center")
    variance = np.var(z, axis=0)
    d = np.sum((z-c)**2, axis=1)
    return dict(per_dimension_variance=variance.tolist(), total_variance=float(variance.sum()),
                possible_collapse=bool(variance.sum() < 1e-8),
                distance_std=float(d.std()), center=c.tolist(),
                caution="Absolute collapse heuristic depends on embedding/score scale; inspect learning curves and distributions.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline scores with per-model normal-validation calibration")
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-fpr", type=float, default=.01)
    parser.add_argument("--final-test", action="store_true")
    parser.add_argument("--freeze", type=Path, required=True,
                        help="results/selection.json; final test must match its recommended candidate")
    args = parser.parse_args()
    freeze = json.loads(args.freeze.read_text())
    with np.load(args.scores, allow_pickle=False) as d:
        candidate, model_id, seed = str(d["candidate"]), str(d["model_id"]), int(d["seed"])
        if candidate not in freeze["candidates"]:
            raise ValueError("Unknown candidate")
        contract_path = args.freeze.parent / candidate / "preprocessing.json"
        contract_sha = hashlib.sha256(contract_path.read_bytes()).hexdigest()
        if str(d["contract_sha256"]) != contract_sha or freeze["candidates"][candidate]["preprocessing_sha256"] != contract_sha:
            raise ValueError("Score contract does not match frozen preprocessing")
        # Row IDs validate exact score ordering against the exported data.
        with np.load(args.freeze.parent / candidate / "baseline.npz", allow_pickle=False) as data:
            if not np.array_equal(d["validation_row_ids"], data["validation_row_ids"]):
                raise ValueError("Validation score row IDs/order mismatch")
            val_scores = _scores(d["validation_scores"])
            yv, av = data["y_validation"], data["attack_validation"]
            if len(val_scores) != len(yv):
                raise ValueError("Validation score count mismatch")
            calibration = calibrate(val_scores[yv == 0], args.target_fpr)
            result = dict(candidate=candidate, model_id=model_id, seed=seed,
                          contract_sha256=contract_sha, calibration=calibration,
                          validation=metrics(val_scores, yv, av, calibration["threshold"]))
            if "validation_embeddings" in d and "center" in d:
                z, center = d["validation_embeddings"], d["center"]
                if len(z) != len(yv) or not np.allclose(val_scores, np.sum((z-center)**2, axis=1), rtol=1e-4, atol=1e-7):
                    raise ValueError("Validation scores disagree with supplied embeddings/center")
                result["embeddings"] = embedding_diagnostics(z, center)
            if args.final_test:
                if candidate != freeze["recommended_candidate"]:
                    raise ValueError("Final test candidate differs from frozen recommendation")
                if not np.array_equal(d["test_row_ids"], data["test_row_ids"]):
                    raise ValueError("Test score row IDs/order mismatch")
                result["test"] = metrics(d["test_scores"], data["y_test"], data["attack_test"], calibration["threshold"])
                # A write-once receipt guards accidental repeated final evaluations.
                receipt = args.freeze.parent / f"final_test_seed_{seed}.json"
                with receipt.open("x") as f:
                    json.dump(dict(model_id=model_id, seed=seed, candidate=candidate,
                                   contract_sha256=contract_sha, result=result), f, indent=2)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
