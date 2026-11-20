from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import platform
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch

from svdd_features.data import load_ids, disjoint_encoded
from svdd_features.evaluation import calibrate
from .common import save_json, sha, identity, array_sha
from .data import prepare
from .evaluation import evaluate, benchmark
from .training import train, load_model, trace_numpy, configure
from .reference import export, verify, load_parameters, numpy_infer
from .quantization import fixed_infer, numerical_ranges
from .reporting import plots, write_report


def golden_indices(labels, attacks, scores, threshold, count, seed):
    """Coverage/threshold cases are integration examples, not a performance sample."""
    count = min(count, len(labels))
    if count < 10:
        raise ValueError("Too few golden cases")
    chosen = set()
    # Preserve family coverage, including benign.
    for family in np.unique(attacks):
        chosen.add(int(np.flatnonzero(attacks == family)[0]))
    if len(chosen) > count:
        raise ValueError("Golden budget cannot cover families")
    for mask in [scores <= threshold, scores > threshold]:
        indices = np.flatnonzero(mask)
        near = indices[np.argsort(np.abs(scores[indices]-threshold), kind="stable")[:min(32, count//4)]]
        for i in near:
            if len(chosen) < count:
                chosen.add(int(i))
    rng = np.random.default_rng(seed)
    pools = [rng.permutation(np.flatnonzero(labels == k)).tolist() for k in [0, 1]]
    while len(chosen) < count and any(pools):
        for pool in pools:
            while pool and pool[-1] in chosen:
                pool.pop()
            if pool and len(chosen) < count:
                chosen.add(int(pool.pop()))
    return np.array(sorted(chosen), dtype=int)


def validate_config(c):
    if c["primary_seed"] not in c["compact_seeds"] or len(set(c["compact_seeds"])) != len(c["compact_seeds"]):
        raise ValueError("Invalid compact seeds/primary")
    t = c["training"]
    if not 1 <= t["min_epochs"] <= t["max_epochs"] or t["patience"] <= 0 or t["regularization"] < 0 or t["learning_rate"] <= 0 or t["batch_size"] <= 0 or t["cpu_threads"] <= 0:
        raise ValueError("Invalid training policy")
    if not 0 < c["target_fpr"] < 1 or c["golden_rows"] < 100:
        raise ValueError("Invalid FPR/golden vector count")
    for q in c["fixed_point"]:
        if not 2 <= q["integer_bits"] < q["bits"] <= 24:
            raise ValueError("Unsupported quantization format")


def execute(config, output, requested_device):
    validate_config(config)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    if (out/"config.json").exists() and json.loads((out/"config.json").read_text()) != config:
        raise ValueError("Output directory contains another configuration; choose a new --output")
    save_json(out/"config.json", config)
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA explicitly requested but unavailable")
    device = "cuda" if requested_device != "cpu" and torch.cuda.is_available() else "cpu"
    configure(config["primary_seed"], config["training"]["cpu_threads"])
    print(f"Software baseline device: {device}; preparing normal-only data", flush=True)
    prepared, validation, ids, fingerprint = prepare(config, out)
    yv, av = validation.Label.to_numpy(np.uint8), validation.Attack.to_numpy(dtype=str)
    train_sources = {str(p): sha(p) for p in [Path("svdd_baseline/model.py"), Path("svdd_baseline/training.py"), Path("svdd_features/preprocessing.py")]}
    experiment_id = identity(dict(config=config, dataset=fingerprint,
        contracts={k: v["contract_sha256"] for k, v in prepared.items()}, training_sources=train_sources))
    if (out/"freeze.json").exists() and json.loads((out/"freeze.json").read_text())["experiment_id"] != experiment_id:
        raise ValueError("Frozen run identity changed; choose a new output directory")
    cpu = platform.processor()
    if Path("/proc/cpuinfo").exists():
        cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")), cpu)
    environment = dict(python=platform.python_version(), platform=platform.platform(),
        cpu=cpu, logical_cpu_count=os.cpu_count(), torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__,
        scipy=scipy.__version__, sklearn=sklearn.__version__, cuda_build=torch.version.cuda,
        cuda_available=torch.cuda.is_available(), selected_device=device,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        installed_packages=subprocess.check_output([os.sys.executable, "-m", "pip", "freeze"], text=True).splitlines(),
        source_sha256={str(p): sha(p) for package in ["svdd_features", "svdd_baseline"] for p in sorted(Path(package).glob("*.py"))})
    save_json(out/"environment.json", environment)
    jobs = [(kind, seed) for kind in ["eligible", "compact"] for seed in config[kind+"_seeds"]]
    records, frozen = {}, {}
    for kind, seed in jobs:
        name, inputs = f"{kind}_seed{seed}", prepared[kind]
        folder = out/name
        model, training = train(folder, inputs["x_train"], inputs["x_validation"][yv == 0], seed, config["training"], device, experiment_id)
        traces = trace_numpy(model, inputs["x_validation"], device)
        calibration = calibrate(traces["scores"][yv == 0], config["target_fpr"])
        calibration.update(checkpoint_sha256=training["checkpoint_sha256"], preprocessing_sha256=inputs["contract_sha256"])
        # An existing receipt must retain exactly the same threshold/model pairing.
        if (folder/"test_receipt.json").exists():
            existing = json.loads((folder/"threshold.json").read_text())
            if existing != calibration:
                raise ValueError("Previously tested model's calibration changed")
        save_json(folder/"threshold.json", calibration)
        np.savez_compressed(folder/"validation_scores.npz", **traces, row_ids=validation.index.to_numpy(), labels=yv, attacks=av)
        parameters, metadata = export(folder/"export", model, inputs["pipe"], calibration["threshold"], training, experiment_id)
        gi = golden_indices(yv, av, traces["scores"], calibration["threshold"], config["golden_rows"], config["data_seed"])
        verify(folder/"export", inputs["x_validation"][gi], {k: v[gi] for k, v in traces.items()}, validation.index.to_numpy()[gi])
        records[name] = dict(configuration=kind, seed=seed, training=training, calibration=calibration,
            validation=evaluate(traces["scores"], yv, av, calibration["threshold"]), export=metadata)
        frozen[name] = dict(checkpoint_sha256=training["checkpoint_sha256"], contract_sha256=inputs["contract_sha256"],
            center=training["center"], threshold=calibration["threshold"], seed=seed, input_dimension=training["input_dimension"],
            validation_row_ids_sha256=array_sha(validation.index.to_numpy()))
        print(f"{name}: trained/restored; references verified; threshold={calibration['threshold']:.9g}", flush=True)
        del model

    primary_name = f"compact_seed{config['primary_seed']}"
    primary_folder = out/primary_name
    primary_parameters, _ = load_parameters(primary_folder/"export")
    primary = prepared["compact"]
    primary_model, _ = load_model(primary_folder, device)
    train_trace = trace_numpy(primary_model, primary["x_train"], device)
    val_trace = trace_numpy(primary_model, primary["x_validation"], device)
    range_study = dict(training=numerical_ranges(primary_parameters, primary["x_train"], train_trace),
                       validation=numerical_ranges(primary_parameters, primary["x_validation"], val_trace),
                       format_choice="Two predeclared formats from config, not optimized on test; suggested ranges include 25% headroom")
    qcalibrations, quantized_validation = {}, {}
    for q in config["fixed_point"]:
        qname = f"Q{q['bits']}_I{q['integer_bits']}"
        qt, details = fixed_infer(primary_parameters, primary["x_validation"], q["bits"], q["integer_bits"])
        qcalibrations[qname] = calibrate(qt["scores"][yv == 0], config["target_fpr"])
        quantized_validation[qname] = dict(arithmetic=details, calibration=qcalibrations[qname],
            original_threshold_metrics=evaluate(qt["scores"], yv, av, primary_parameters["threshold"]),
            recalibrated_metrics=evaluate(qt["scores"], yv, av, qcalibrations[qname]["threshold"]))
    freeze = dict(experiment_id=experiment_id, dataset_sha256=fingerprint, models=frozen,
        primary_model=primary_name, selection_policy="Predeclared compact seed; no attack labels for selection",
        target_fpr=config["target_fpr"], fixed_point_formats=config["fixed_point"],
        quantized_validation_calibrations=qcalibrations, test_row_ids_sha256=array_sha(ids["test"]))
    if (out/"freeze.json").exists() and json.loads((out/"freeze.json").read_text()) != freeze:
        raise ValueError("Freeze manifest differs; refusing to change previously frozen choices")
    save_json(out/"freeze.json", freeze)
    print("All floating-point models and numerical-study choices frozen. Loading test rows for final evaluation.", flush=True)
    started = time.perf_counter()
    test = load_ids(Path(config["data"]), {"test": ids["test"]}, config["chunk_size"])["test"]
    loading_test_seconds = time.perf_counter()-started
    yt, at = test.Label.to_numpy(np.uint8), test.Attack.to_numpy(dtype=str)
    test_inputs = {}
    for kind, inputs in prepared.items():
        started = time.perf_counter()
        test_inputs[kind] = inputs["pipe"].transform(test)
        inputs["test_preprocessing_seconds"] = time.perf_counter()-started
        # Report projection collisions; preserve the same original evaluation rows.
        inputs["test_vectors_also_in_normal_training"] = int((~disjoint_encoded([inputs["x_train"]], test_inputs[kind])).sum())
    for name, record in records.items():
        folder, x = out/name, test_inputs[record["configuration"]]
        receipt = folder/"test_receipt.json"
        if receipt.exists():
            old = json.loads(receipt.read_text())
            if old["freeze_sha256"] != sha(out/"freeze.json") or old["scores_sha256"] != sha(folder/"test_scores.npz") or old["evaluation_sha256"] != sha(folder/"evaluation.json"):
                raise ValueError("Test receipt/score identity mismatch")
            result = json.loads((folder/"evaluation.json").read_text())
        else:
            model, _ = load_model(folder, device)
            trace = trace_numpy(model, x, device)
            result = evaluate(trace["scores"], yt, at, record["calibration"]["threshold"])
            np.savez_compressed(folder/"test_scores.npz", **trace, row_ids=test.index.to_numpy(), labels=yt, attacks=at)
            save_json(folder/"evaluation.json", result)
            save_json(receipt, dict(experiment_id=experiment_id, freeze_sha256=sha(out/"freeze.json"),
                checkpoint_sha256=record["training"]["checkpoint_sha256"], scores_sha256=sha(folder/"test_scores.npz"),
                evaluation_sha256=sha(folder/"evaluation.json")))
        record["test"] = result
        model, _ = load_model(folder, "cpu")
        record["benchmark"] = benchmark(model, prepared[record["configuration"]]["x_validation"], config["benchmark"])
        save_json(folder/"benchmark.json", record["benchmark"])
        print(f"{name}: TEST AUROC={result['roc_auc']:.6f} AP={result['average_precision']:.6f} recall={result['recall']:.6f} FPR={result['measured_fpr']:.6f}", flush=True)

    with np.load(primary_folder/"test_scores.npz", allow_pickle=False) as z:
        expected = {k: z[k] for k in ["h1", "h2", "embedding", "distance_terms", "scores"]}
    gi = golden_indices(yt, at, expected["scores"], primary_parameters["threshold"], config["golden_rows"], config["data_seed"])
    reference_report = verify(primary_folder/"export", test_inputs["compact"][gi], {k: v[gi] for k, v in expected.items()}, test.index.to_numpy()[gi])
    test.iloc[gi][primary["pipe"].contract["raw_columns"]].to_csv(primary_folder/"export"/"golden_raw.csv", index=False, float_format="%.17g")
    labels = test.iloc[gi][["Label", "Attack"]].copy()
    labels.insert(0, "row_id", labels.index)
    labels["absolute_threshold_margin"] = np.abs(expected["scores"][gi]-primary_parameters["threshold"])
    labels.to_csv(primary_folder/"export"/"golden_labels.csv", index=False)
    golden_read = pd.read_csv(primary_folder/"export"/"golden_raw.csv", float_precision="round_trip")
    np.testing.assert_array_equal(primary["pipe"].transform(golden_read), test_inputs["compact"][gi])
    range_study["test_observed_after_freeze"] = numerical_ranges(primary_parameters, test_inputs["compact"], expected)
    quantized = {}
    for q in config["fixed_point"]:
        name = f"Q{q['bits']}_I{q['integer_bits']}"
        trace, details = fixed_infer(primary_parameters, test_inputs["compact"], q["bits"], q["integer_bits"])
        delta = trace["scores"]-expected["scores"]
        # Parameter saturation is reported separately from per-flow data saturation.
        quantized[name] = dict(arithmetic=details, score_mae=float(np.mean(np.abs(delta))),
            max_absolute_score_error=float(np.max(np.abs(delta))),
            score_rmse=float(np.sqrt(np.mean(delta**2))),
            decision_disagreements=int(np.count_nonzero((trace["scores"] > primary_parameters["threshold"]) != (expected["scores"] > primary_parameters["threshold"]))),
            data_saturation_count=sum(v["saturated"] for k, v in details["saturation"].items() if not k.startswith("weights") and k != "center"),
            original_threshold_metrics=evaluate(trace["scores"], yt, at, primary_parameters["threshold"]),
            recalibrated_metrics=evaluate(trace["scores"], yt, at, qcalibrations[name]["threshold"]),
            recalibrated_threshold=qcalibrations[name])
    save_json(out/"numerical_study.json", dict(ranges=range_study, validation=quantized_validation, test=quantized))
    timings = dict(loading_test_csv_seconds=loading_test_seconds, configurations={k: {
        v: x[v] for v in ["preprocessing_seconds", "transformed_rows", "test_preprocessing_seconds", "test_vectors_also_in_normal_training"]} for k, x in prepared.items()},
        methodology="Preprocessing times include transforms only after raw frames are loaded; CSV loading separately measured. No FPGA throughput claim.")
    save_json(out/"preprocessing_timings.json", timings)
    compact_records = [v for v in records.values() if v["configuration"] == "compact"]
    aggregated = {k: dict(mean=float(np.mean([r["test"][k] for r in compact_records])),
        std=float(np.std([r["test"][k] for r in compact_records]))) for k in ["roc_auc", "average_precision", "precision", "recall", "f1", "measured_fpr"]}
    summary = dict(experiment_id=experiment_id, config=config, models=records,
        dataset=json.loads((out/"dataset_audit.json").read_text()), split=json.loads((out/"split_manifest.json").read_text()),
        compact_seed_summary=aggregated, primary_reference_verification=reference_report, quantized=quantized)
    save_json(out/"summary.json", summary)
    plots(out, records)
    write_report(out, summary)
    from .diagnostics import audit as audit_frozen_models
    audit_frozen_models(out)
    save_json(out/"completion.json", dict(experiment_id=experiment_id, status="complete", summary_sha256=sha(out/"summary.json"),
        primary_export_metadata_sha256=sha(primary_folder/"export"/"metadata.json"),
        diagnostics_sha256=sha(out/"diagnostics.json"),
        source_sha256={str(p): sha(p) for package in ["svdd_features", "svdd_baseline"] for p in sorted(Path(package).glob("*.py"))}))
    print(f"Completed software baseline: {out/'report.md'}", flush=True)
    return summary


def main():
    p = argparse.ArgumentParser(description="Train, freeze, evaluate and export the Deep SVDD baseline")
    p.add_argument("--config", type=Path, default=Path("configs/software.json"))
    p.add_argument("--output", type=Path, default=Path("runs/software"))
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = p.parse_args()
    execute(json.loads(args.config.read_text()), args.output, args.device)


if __name__ == "__main__":
    main()
