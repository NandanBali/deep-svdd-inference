"""Command-line reproducible inspection, split, selection and integration export."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
import scipy
import sklearn

from .acquire import acquire, digest
from .data import inspect, chronological_split, sample_ids, load_ids, disjoint_encoded
from .preprocessing import Preprocessor
from .schema import audit, SOURCES
from .selection import track_a, track_b, make_candidates


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def run(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if list(out.glob("final_test_seed_*.json")):
        raise ValueError("Final test already evaluated here. Use a new output directory to preserve the frozen run.")
    path = acquire(Path.cwd()) if args.download else args.data
    if path is None or not path.exists():
        raise ValueError("Supply --download or --data /path/to/audited-v3.csv")
    print("Inspecting full file in bounded chunks", flush=True)
    meta, inspection = inspect(path, args.chunk_size)
    meta["family_names"] = inspection["family_codes"]
    partition, split = chronological_split(meta, args.train_fraction, args.validation_fraction, args.gap_ms)
    ids = sample_ids(partition, args.sample_rows, args.seed)
    # Seal test raw data until the provisional choice is frozen. Training/validation
    # row selection is independent of labels, and of any test values/metrics.
    frames = load_ids(path, {k: v for k, v in ids.items() if k != "test"}, args.chunk_size)
    train, val = frames["train"], frames["validation"]
    normal = train.loc[train.Label == 0]
    if len(normal) < 100 or len(train.Label.unique()) < 2:
        raise ValueError("Insufficient normal/training attack samples for selection")
    np.savez_compressed(out / "splits.npz", partition=partition, **{f"sample_{k}_ids": v for k, v in ids.items()})
    inspection["csv_sha256"] = digest(path)
    dump(out / "inspection.json", inspection)
    split["bounded_sample"] = dict(limit_per_partition=args.sample_rows, seed=args.seed,
        algorithm="numpy PCG64 choice without replacement of eligible row IDs; sorted output; no class balancing",
        sizes={k: len(v) for k, v in ids.items()}, normal_training_rows=len(normal))
    dump(out / "splits.json", split)
    print("Track A: normal-training statistics", flush=True)
    a = track_a(train, args.correlation, args.modal_limit)
    for name, matrix in a.pop("matrices").items():
        matrix.to_csv(out / f"A_{name}.csv", float_format="%.10g")
    dump(out / "track_a.json", a)
    print("Track B: training-label-assisted MI and mRMR", flush=True)
    b = track_b(train, args.seed, args.mi_rows, penalty=args.mrmr_penalty)
    dump(out / "track_b.json", b)
    pd.DataFrame(b["ranking"]).to_csv(out / "B_mi_ranking.csv", index=False)
    choices = make_candidates(train, a, b)
    # Freeze using extraction/preprocessing scope and normal statistics only.
    # A baseline does not exist; no validation attack labels or test metrics used.
    selection = dict(recommended_candidate="A_compact", recommendation_status="provisional: no Deep SVDD baseline available",
        recommendation_basis="Bounded hardware extraction scope, clear original measurements, normal-training variability/redundancy; no detection claim",
        validation_attack_labels_used_for_selection=False, test_used_for_selection=False,
        baseline_available=False, deep_svdd_metrics=None, target_fpr=args.target_fpr,
        baseline_policy=dict(architecture="d -> 16 -> 8, bias-free; same activations/training budget for all candidates",
            training_population="x_train contains only benign training rows", suggested_seeds=[7, 17, 27],
            center_and_threshold="Separate for every candidate and seed; calibrate threshold using normal validation scores only",
            selection="Compare validation ROC-AUC, PR-AUC and recall/FPR + per-attack recall; disclose attack-validation selection; freeze before final test",
            collapse="Export validation embeddings and center; inspect dimension variances, score distributions and training history"),
        candidates=choices)
    pipes, validation_exports = {}, {}
    for name, item in choices.items():
        if not item["features"]:
            raise ValueError(f"Empty candidate: {name}")
        folder = out / name
        folder.mkdir(exist_ok=True)
        pipe = Preprocessor.fit(normal, item["features"])
        pipe.save(folder / "preprocessing.json")
        restored = Preprocessor.load(folder / "preprocessing.json")
        xn, xt, xv = pipe.transform(normal), pipe.transform(train), pipe.transform(val)
        assert np.array_equal(xv, restored.transform(val[val.columns[::-1]]))
        # All mixed training inputs, including training attacks used by Track B,
        # protect against selected/encoded vector duplication in validation.
        mask = disjoint_encoded([xt], xv)
        item.update(raw_feature_count=len(item["features"]), encoded_input_dimension=xn.shape[1],
                    normal_training_rows=len(xn), validation_rows=int(mask.sum()),
                    validation_encoded_duplicate_purge=int((~mask).sum()),
                    preprocessing_sha256=digest(folder / "preprocessing.json"),
                    reason_each_feature={c: dict(contribution=next(r["meaning"] for r in audit([c])),
                        selection_reason=(a["removed"].get(c, "normal-training variable, retained after redundancy analysis") if item["track"] == "A" else "training-label relevance/redundancy criterion")) for c in item["features"]})
        pipes[name] = (pipe, xn, xt)
        validation_exports[name] = (xv[mask], val.loc[mask])
    # Different encodings create different collision sets. Use their intersection
    # for validation comparison so future candidate metrics share identical rows.
    common_validation = val.index.to_numpy()
    for _, vframe in validation_exports.values():
        common_validation = np.intersect1d(common_validation, vframe.index.to_numpy())
    if not len(common_validation):
        raise ValueError("No common disjoint validation rows remain across candidates")
    for name, (xv, vframe) in validation_exports.items():
        common_mask = vframe.index.isin(common_validation)
        validation_exports[name] = (xv[common_mask], vframe.loc[common_mask])
        choices[name]["validation_rows_before_common_intersection"] = len(vframe)
        choices[name]["validation_rows"] = int(common_mask.sum())
    selection["common_validation_rows"] = len(common_validation)
    selection["validation_comparison_policy"] = "Same common raw row IDs for all three candidates after union of per-candidate encoded-collision purges. Do not compare per-candidate filtered populations."
    dump(out / "selection.json", selection)  # choice frozen BEFORE loading test raw features
    print("Provisional recommendation frozen; exporting final test inputs without scoring", flush=True)
    test = load_ids(path, {"test": ids["test"]}, args.chunk_size)["test"]
    verification = {}
    for name, (pipe, xn, xt) in pipes.items():
        folder = out / name
        xv, vframe = validation_exports[name]
        arrays = dict(
            x_train=xn, train_row_ids=normal.index.to_numpy(),
            x_validation=xv, y_validation=vframe.Label.to_numpy(np.uint8),
            attack_validation=vframe.Attack.to_numpy(dtype=str), validation_row_ids=vframe.index.to_numpy())
        final = name == selection["recommended_candidate"]
        if final:
            x_test = pipe.transform(test)
            test_mask = disjoint_encoded([xt, xv], x_test)
            tframe, x_test = test.loc[test_mask], x_test[test_mask]
            arrays.update(x_test=x_test, y_test=tframe.Label.to_numpy(np.uint8),
                attack_test=tframe.Attack.to_numpy(dtype=str), test_row_ids=tframe.index.to_numpy())
        np.savez_compressed(folder / "baseline.npz", **arrays)
        # Golden vectors from TRAINING data: no test examples inspected for recommendation.
        example = pd.concat([normal.head(4), train.loc[train.Label == 1].groupby("Attack", sort=True).head(1)])
        raw = example[item_columns := pipe.contract["raw_columns"]].copy()
        raw.to_csv(folder / "example_raw.csv", index=False, float_format="%.17g")
        labels = example[["Label", "Attack"]].copy()
        labels.insert(0, "row_id", example.index)
        labels.to_csv(folder / "example_labels.csv", index=False)
        x_example = pipe.transform(example)
        pd.DataFrame(x_example, columns=pipe.contract["output_names"]).to_csv(folder / "example_vectors.csv", index=False, float_format="%.9g")
        raw_roundtrip = pd.read_csv(folder / "example_raw.csv", float_precision="round_trip")
        vector_roundtrip = pd.read_csv(folder / "example_vectors.csv").to_numpy(np.float32)
        np.testing.assert_array_equal(Preprocessor.load(folder / "preprocessing.json").transform(raw_roundtrip), x_example)
        np.testing.assert_array_equal(vector_roundtrip, x_example)
        assert set(item_columns).isdisjoint({"Label", "Attack", "IPV4_SRC_ADDR", "IPV4_DST_ADDR", "FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS", "DNS_QUERY_ID"})
        assert np.isfinite(xn).all() and np.isfinite(xv).all()
        assert disjoint_encoded([xt], xv).all()
        if final:
            assert np.isfinite(x_test).all() and disjoint_encoded([xt, xv], x_test).all()
        verification[name] = dict(save_reload_exact=True, column_reordering_exact=True,
            labels_excluded=True, golden_raw_and_exported_vectors_exact=True,
            finite_float32=True, encoded_partition_disjoint=True,
            test_rows_exported=len(tframe) if final else 0,
            test_encoded_duplicate_purge=int((~test_mask).sum()) if final else None,
            test_metrics_computed=False,
            baseline_array_sha256={k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k, v in arrays.items()})
    dump(out / "verification.json", verification)
    full_audit = audit(inspection["columns"])
    for item in full_audit:
        c = item["name"]
        item["quality"] = inspection["column_quality"][c]
        item["retained_in"] = [n for n, subset in choices.items() if c in subset["features"]]
        item["selection_note"] = a["removed"].get(c, "See candidate policy and Track B ranking; excluded metadata/ambiguous fields never enter candidates" if not item["candidate"] else "Within candidate pool; bounded subset budget applies")
    dump(out / "feature_audit.json", full_audit)
    dump(out / "run.json", dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        environment=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                         scipy=scipy.__version__, sklearn=sklearn.__version__),
        csv_sha256=inspection["csv_sha256"], sources=SOURCES,
        source_sha256={str(p): digest(p) for p in sorted(Path("svdd_features").glob("*.py"))}))
    write_report(out, inspection, split, a, b, selection, verification)
    make_plot(out, a, b, selection)
    print(json.dumps({n: dict(raw=v["raw_feature_count"], encoded=v["encoded_input_dimension"], features=v["features"]) for n, v in choices.items()}, indent=2), flush=True)


def write_report(out, inspection, split, a, b, selection, verification):
    # Keep the reproducible numeric findings generated from artifacts; interpretation
    # and developer handoff are maintained in README.md and REPORT.md.
    lines = ["# Generated run findings", "", "Recommendation: **A_compact (provisional; baseline absent)**.", "",
        f"Full CSV: {inspection['rows']:,} flows, 53 raw features, 2 labels; SHA-256 `{inspection['csv_sha256']}`.",
        f"Exact duplicate rows (hash-based): {inspection['exact_duplicate_rows']:,}; duplicate eligible raw feature vectors: {inspection['duplicate_candidate_vectors']:,}; conflicting-label feature groups: {inspection['candidate_vectors_with_conflicting_binary_labels']:,}.", "",
        "## Full-file family distribution", "", "| Family | Rows |", "|---|---:|"]
    lines += [f"| {k} | {v:,} |" for k, v in inspection["attack_distribution"].items()]
    lines += ["", "## Split and sampling", "", f"Cutoffs (epoch ms): {split['cutoffs_ms']}; symmetric embargo: {split['embargo_ms']} ms.",
        f"Time/boundary purges: {split['time_or_boundary_purged_rows']:,}; cross-partition raw-vector duplicate purges: {split['cross_partition_duplicate_rows_purged']:,}.", "",
        "| Partition | Eligible rows | Uniform bounded sample | Normal / attack in eligible partition |", "|---|---:|---:|---|"]
    for name, info in split["partitions"].items():
        lines.append(f"| {name} | {info['rows']:,} | {split['bounded_sample']['sizes'][name]:,} | {info['class_distribution']} |")
    lines += ["", "## Track A: no attack samples for ranking", "",
        f"{a['normal_rows']:,} normal training rows; absolute Pearson OR Spearman threshold {a['correlation_threshold']}; modal limit {a['modal_limit']}. Covariance uses training-fitted log/quantile/min-max scaling.",
        "Normal variability/redundancy establishes distinguishable inputs, not anomaly-detection performance.", "",
        "Removed inputs:"]
    lines += [f"- `{c}`: {why}." for c, why in a["removed"].items()]
    lines += ["", "## Track B: uses attack training labels", "",
        f"MI sample: {b['rows']:,} training rows without balancing; class counts {b['class_distribution']}; seed {b['seed']}.",
        b["mi_estimator"]+".", b["mrmr_formula"]+".", b["redundancy_estimator"], "",
        "| Feature | MI (nats) | Quantile-bin relevance (nats) |", "|---|---:|---:|"]
    lines += [f"| {r['feature']} | {r['mi_nats']:.6f} | {r['binned_mi_nats']:.6f} |" for r in b["ranking"][:15]]
    lines += ["", "## Three bounded candidates", "", "| Candidate | Track | Raw | Encoded | Features |", "|---|---|---:|---:|---|"]
    for name, item in selection["candidates"].items():
        lines.append(f"| {name} | {item['track']} | {item['raw_feature_count']} | {item['encoded_input_dimension']} | {', '.join(item['features'])} |")
    lines += ["", "## Baseline status and freeze", "",
        "No working Deep SVDD evaluator or model existed in the repository. No neural training was added. ROC-AUC, PR-AUC, recall/FPR, per-attack recall, score distributions and embedding-collapse results are **not available**.",
        "No attack validation labels were used to choose the recommendation. The provisional choice was saved before test raw features were loaded for the handoff export. No test scores or metrics were computed.",
        f"After subset selection, clipping and encoding can create new duplicate vectors. Validation collisions against all mixed training inputs were removed per candidate, then intersected into {selection['common_validation_rows']:,} identical validation row IDs for fair comparison. This substantially changes the population. Test collisions were purged against training/retained validation for the frozen recommendation only; other test arrays are not exported.", "",
        "| Candidate | Validation vectors purged | Test vectors purged | Test rows exported |", "|---|---:|---:|---:|"]
    for name, item in selection["candidates"].items():
        v = verification[name]
        test_purge = f"{v['test_encoded_duplicate_purge']:,}" if v["test_encoded_duplicate_purge"] is not None else "not exported"
        lines.append(f"| {name} | {item['validation_encoded_duplicate_purge']:,} | {test_purge} | {v['test_rows_exported']:,} |")
    lines += ["", "## Limitations and handoff", "", split["limitations"],
        "All full-file class/quality counts are descriptive auditing, not selection/evaluation evidence. Within-partition multiplicity and conflicting labels are retained; duplicates crossing time partitions are purged from the later partition.",
        "Fit/ranking/model exports use bounded uniform samples. Preprocessing and Track A use only normal training; Track B uses mixed training. Categorical MI considers full nominal codes whereas model encoding caps learned nominal categories; this information loss must be validated.",
        "Normal q0.001/q0.999 are scaling anchors, with training-derived saturation bounds extending the encoded range to [-1,4]. This preserves bounded contrast above the normal upper quantile (e.g. TTL 252 versus 31), although far-tail distinctions still saturate. This v1 policy needs baseline validation. One-hot other is present only if normal frequency is in [0.0005,0.9995], else unknowns map to all zeros. Flag fields expand to all eight specified bits.",
        "Consume `A_compact/preprocessing.json` for exact raw order, formula, encoding, missing behavior and dimension; `baseline.npz` for baseline arrays; `example_raw.csv` + `example_vectors.csv` + `example_labels.csv` for integration; `feature_audit.json` for extraction state/arithmetic/unknown definitions.",
        "Extraction costs are ordinal engineering judgments, separate from software preprocessing and MLP dimension. No LUT/DSP/BRAM/resource estimates or throughput claims without synthesis.",
        "See root `REPORT.md` for interpretation and `README.md` for the single reproduction command and baseline score interface.", ""]
    (out / "findings.md").write_text("\n".join(lines))


def make_plot(out, a, b, selection):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    columns = selection["candidates"]["A_compact"]["features"]
    corr = pd.read_csv(out / "A_pearson.csv", index_col=0).loc[columns, columns]
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), layout="constrained")
    im = axes[0].imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    labels = [c.replace("FLOW_DURATION_MILLISECONDS", "DURATION_MS").replace("TCP_WIN_MAX_", "WIN_MAX_") for c in columns]
    axes[0].set_xticks(range(len(columns)), labels, rotation=60, ha="right", fontsize=8)
    axes[0].set_yticks(range(len(columns)), labels, fontsize=8)
    axes[0].set_title("Track A: retained-input Pearson correlation\nNormal training only; not detection performance")
    fig.colorbar(im, ax=axes[0], shrink=.65)
    ranking = list(reversed(b["ranking"][:12]))
    axes[1].barh([r["feature"] for r in ranking], [r["mi_nats"] for r in ranking], color="#326e8d")
    axes[1].set_xlabel("Mutual information (nats)")
    axes[1].tick_params(axis="y", labelsize=8)
    axes[1].set_title("Track B: label-assisted MI ranking\nMixed training sample; not Deep SVDD performance")
    fig.savefig(out / "training_diagnostics.png", dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--download", action="store_true", help="Download/verify pinned official v3 BagIt payload")
    source.add_argument("--data", type=Path, help="Audited v3 CSV (schema validated; checksum recorded)")
    p.add_argument("--output", type=Path, default=Path("results"))
    p.add_argument("--chunk-size", type=int, default=100_000)
    p.add_argument("--sample-rows", type=int, default=100_000)
    p.add_argument("--mi-rows", type=int, default=30_000)
    p.add_argument("--seed", type=int, default=20261110)
    p.add_argument("--train-fraction", type=float, default=.6)
    p.add_argument("--validation-fraction", type=float, default=.2)
    p.add_argument("--gap-ms", type=int, default=60_000)
    p.add_argument("--correlation", type=float, default=.95)
    p.add_argument("--modal-limit", type=float, default=.9995)
    p.add_argument("--mrmr-penalty", type=float, default=.1)
    p.add_argument("--target-fpr", type=float, default=.01)
    args = p.parse_args()
    if args.chunk_size <= 0 or args.mi_rows < 10 or not 0 < args.correlation <= 1 or not .5 < args.modal_limit <= 1 or not 0 < args.target_fpr < 1 or args.mrmr_penalty < 0:
        p.error("Invalid estimation/splitting parameter")
    run(args)


if __name__ == "__main__":
    main()
