from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from svdd_features.data import inspect, chronological_split, sample_ids, load_ids
from svdd_features.preprocessing import Preprocessor
from svdd_features.schema import candidates, audit, feature_info
from svdd_features.selection import track_a, PRIORITY
from .common import sha, save_json, array_sha


def prepare(config, out):
    path = Path(config["data"])
    if not path.exists():
        raise FileNotFoundError(f"Dataset missing: {path}. Set data in the configuration; no dataset substitution is allowed.")
    fingerprint = sha(path)
    old = Path("results")
    reusable = not config["synthetic"] and (old/"inspection.json").exists() and (old/"splits.npz").exists()
    if reusable:
        old_audit = json.loads((old/"inspection.json").read_text())
        old_split = json.loads((old/"splits.json").read_text())
        reusable = old_audit["csv_sha256"] == fingerprint and all(old_split[k] == config[k] for k in ["train_fraction", "validation_fraction"]) and old_split["embargo_ms"] == config["gap_ms"]
    if reusable:
        dataset_audit, split = old_audit, old_split
        with np.load(old/"splits.npz", allow_pickle=False) as z:
            partition = z["partition"]
    else:
        meta, dataset_audit = inspect(path, config["chunk_size"])
        meta["family_names"] = dataset_audit["family_codes"]
        partition, split = chronological_split(meta, config["train_fraction"], config["validation_fraction"], config["gap_ms"])
        dataset_audit["csv_sha256"] = fingerprint
    ids = sample_ids(partition, config["sample_rows"], config["data_seed"])
    cache = out/"data"
    cache.mkdir(exist_ok=True)
    np.savez_compressed(cache/"split_indices.npz", partition=partition, **{k+"_row_ids": v for k, v in ids.items()})
    started = time.perf_counter()
    frames = load_ids(path, {k: v for k, v in ids.items() if k != "test"}, config["chunk_size"])
    loading_seconds = time.perf_counter()-started
    train, validation = frames["train"], frames["validation"]
    normal = train.loc[train.Label == 0]
    if len(normal) < 100 or set(validation.Label.unique()) != {0, 1}:
        raise ValueError("Insufficient normal training or mixed validation data")
    assert set(train.index).isdisjoint(validation.index)
    assert set(ids["test"]).isdisjoint(set(train.index) | set(validation.index))
    # This selector sees normal records only, including a constant zero Label.
    a = track_a(normal)
    for name, matrix in a.pop("matrices").items():
        matrix.to_csv(cache/f"normal_{name}.csv")
    save_json(cache/"compact_selection.json", a)
    compact = [c for c in PRIORITY if c in a["retained_numeric"] and feature_info(c)["extraction_cost_ordinal"] <= 2][:8]
    feature_sets = {"eligible": candidates(list(normal)), "compact": compact}
    prepared, table = {}, []
    for name, columns in feature_sets.items():
        pipe = Preprocessor.fit(normal, columns, **config["preprocessing"])
        pipe.save(cache/f"{name}_preprocessing.json")
        started = time.perf_counter()
        xn, xv = pipe.transform(normal), pipe.transform(validation)
        preprocessing_seconds = time.perf_counter()-started
        np.testing.assert_array_equal(xv, Preprocessor.load(cache/f"{name}_preprocessing.json").transform(validation[validation.columns[::-1]]))
        assert np.isfinite(xn).all() and np.isfinite(xv).all()
        np.savez_compressed(cache/f"{name}_development.npz", x_train=xn, train_row_ids=normal.index.to_numpy(),
            x_validation=xv, validation_row_ids=validation.index.to_numpy(), y_validation=validation.Label.to_numpy(np.uint8),
            attack_validation=validation.Attack.to_numpy(dtype=str))
        prepared[name] = dict(pipe=pipe, x_train=xn, x_validation=xv,
            contract_sha256=sha(cache/f"{name}_preprocessing.json"),
            preprocessing_seconds=preprocessing_seconds, transformed_rows=len(xn)+len(xv))
        entries = {e["raw_column"]: e for e in pipe.contract["entries"]}
        for info in audit(list(train)):
            c = info["name"]
            e = entries.get(c)
            table.append(dict(configuration=name, feature=c, meaning=info["meaning"], units=info["units"],
                included=e is not None, reason=("all audited eligible fields" if name == "eligible" else "normal-only variance/redundancy plus cost priority") if e else a["removed"].get(c, "excluded metadata/identifier/ambiguous statistic" if not info["candidate"] else "outside compact feature budget"),
                encoding_and_transformation=e, input_indices=list(range(e["offset"], e["offset"]+len(e["output_names"]))) if e else [],
                extraction_cost_ordinal=info["extraction_cost_ordinal"], extraction=info["per_flow_state"],
                arithmetic=info["update_operations"], hardware_concerns=info["unknowns"]))
    split = dict(split)
    split["software_sample"] = dict(seed=config["data_seed"], cap_per_partition=config["sample_rows"],
        sizes={k: len(v) for k, v in ids.items()}, normal_training_rows=len(normal),
        normal_training_row_ids_sha256=array_sha(normal.index.to_numpy()),
        projected_input_collision_policy="Keep evaluation rows after the pre-fit raw-vector split purge. No feature-dependent encoded collision purge; same evaluation population for both configurations.",
        training_attack_labels_used=False, supervised_feature_selection=False,
        validation_attack_labels_used_for_selection=False,
        loading_development_csv_seconds=loading_seconds)
    dataset_audit["experiment_data_kind"] = "synthetic smoke fixture" if config["synthetic"] else "real NF-UNSW-NB15-v3"
    save_json(out/"dataset_audit.json", dataset_audit)
    save_json(out/"split_manifest.json", split)
    save_json(out/"feature_table.json", table)
    pd.DataFrame([{k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()} for row in table]).to_csv(out/"feature_table.csv", index=False)
    return prepared, validation, ids, fingerprint
