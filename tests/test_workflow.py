"""Synthetic regression fixtures are never reported as dataset/model results."""
from __future__ import annotations

from argparse import Namespace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

from svdd_features.data import chronological_split, sample_ids, disjoint_encoded, row_hash
from svdd_features.preprocessing import Preprocessor
from svdd_features.evaluation import calibrate, metrics, embedding_diagnostics
from svdd_features.schema import expected_columns, feature_info
from svdd_features.selection import track_a, track_b
from svdd_features.workflow import run


@pytest.fixture
def frame():
    rng = np.random.default_rng(12)
    n = 900
    data = {}
    for c in sorted(expected_columns()):
        info = feature_info(c)
        if c == "Attack":
            data[c] = np.where(np.arange(n) % 7 == 0, "TestAttack", "Benign")
        elif c == "Label":
            data[c] = (np.arange(n) % 7 == 0).astype(float)
        elif info["kind"] == "address":
            data[c] = [f"10.0.{i//255}.{i%255}" for i in range(n)]
        elif c == "FLOW_START_MILLISECONDS":
            data[c] = 1420000000000+np.arange(n)*1000.
        elif c == "FLOW_END_MILLISECONDS":
            data[c] = 1420000000000+np.arange(n)*1000.+10
        elif info["kind"] == "port":
            data[c] = rng.choice([0, 80, 443, 4000, 55000], n).astype(float)
        elif c == "PROTOCOL":
            data[c] = rng.choice([1, 6, 17], n).astype(float)
        elif info["kind"] == "flags":
            data[c] = rng.choice([0, 2, 16, 18, 24], n).astype(float)
        elif info["kind"] == "categorical":
            data[c] = rng.integers(0, 4, n).astype(float)
        elif c in {"MIN_TTL", "MAX_TTL"}:
            data[c] = rng.choice([31, 64, 128, 252], n).astype(float)
        else:
            data[c] = rng.integers(0, 10000, n).astype(float)
    return pd.DataFrame(data)


def test_contract_roundtrip_order_and_label_independence(frame, tmp_path):
    cols = ["IN_BYTES", "PROTOCOL", "L4_DST_PORT", "TCP_FLAGS"]
    p = Preprocessor.fit(frame.loc[frame.Label == 0], cols)
    p.save(tmp_path / "p.json")
    restored = Preprocessor.load(tmp_path / "p.json")
    expected = p.transform(frame)
    altered = frame[frame.columns[::-1]].copy()
    altered["Label"] = 3
    altered["Attack"] = "poison"
    altered["IPV4_SRC_ADDR"] = "poison"
    np.testing.assert_array_equal(restored.transform(altered), expected)
    assert expected.dtype == np.float32 and expected.flags["C_CONTIGUOUS"]
    assert np.isfinite(expected).all()
    for forbidden in ["Label", "Attack", "IPV4_SRC_ADDR", "DNS_QUERY_ID", "FLOW_START_MILLISECONDS"]:
        with pytest.raises(ValueError, match="Forbidden"):
            Preprocessor.fit(frame, [forbidden])
    with pytest.raises(ValueError, match="Missing required"):
        p.transform(frame.drop(columns="IN_BYTES"))
    d = json.loads((tmp_path / "p.json").read_text())
    d["raw_columns"][0] = "Label"
    (tmp_path / "bad.json").write_text(json.dumps(d))
    with pytest.raises(ValueError):
        Preprocessor.load(tmp_path / "bad.json")


def test_missing_domain_and_clipping(frame):
    cols = ["IN_BYTES", "MIN_TTL", "PROTOCOL", "L4_DST_PORT", "TCP_FLAGS"]
    p = Preprocessor.fit(frame, cols)
    edge = frame.iloc[:8].copy()
    edge["IN_BYTES"] = [np.nan, np.inf, -1, "bad", 0, 1e300, 3, 50]
    edge["MIN_TTL"] = [np.nan, 256, -1, np.inf, 0, 255, 31, 31]
    edge["PROTOCOL"] = [np.nan, 300, -1, 17.5, 6, 255, 1, 17]
    edge["L4_DST_PORT"] = [0, 1023, 1024, 49151, 49152, 65535, 65536, -1]
    edge["TCP_FLAGS"] = [np.nan, np.inf, -1, 256, 0, 255, 2, 1]
    x = p.transform(edge)
    assert np.isfinite(x).all() and (x >= -1).all() and (x <= 4).all()
    np.testing.assert_array_equal(x[0, :2], x[1, :2])
    assert x[5, 0] == 4
    flags = p.contract["entries"][-1]
    np.testing.assert_array_equal(x[5, flags["offset"]:], np.ones(8))
    np.testing.assert_array_equal(x[:4, flags["offset"]:], np.zeros((4, 8)))


def test_ports_and_protocol_are_nominal(frame):
    p = Preprocessor.fit(frame, ["L4_DST_PORT", "PROTOCOL"])
    f = frame.iloc[:4].copy()
    f["L4_DST_PORT"] = [80, 81, 55000, 55001]
    f["PROTOCOL"] = [6, 6, 17, 17]
    x = p.transform(f)
    np.testing.assert_array_equal(x[0], x[1])
    np.testing.assert_array_equal(x[2], x[3])
    assert not np.array_equal(x[0], x[2])
    assert p.contract["entries"][0]["encoding"] == "one_hot"


def test_normal_only_scaling_preserves_bounded_ttl_contrast():
    train = pd.DataFrame({"MIN_TTL": [0]*10+[31]*990})
    p = Preprocessor.fit(train, ["MIN_TTL"])
    x = p.transform(pd.DataFrame({"MIN_TTL": [31, 64, 252, 255]}))[:, 0]
    assert x[0] == 1 and x[1] > x[0] and x[2] > x[0]
    assert x[-1] == 4


def test_time_split_purges_crossing_and_duplicate_vectors():
    n = 100
    start = np.arange(n)*1000.+1_420_000_000_000
    end = start+10
    h = np.arange(n, dtype=np.uint64)
    h[90] = h[10]  # repeated candidate vector despite a different time/address
    end[15] = start[85]  # flow spans both time boundaries
    meta = dict(start=start, end=end, feature_hash=h,
                label=(np.arange(n) % 7 == 0).astype(np.uint8), family=np.zeros(n, np.uint8))
    p, info = chronological_split(meta, gap_ms=100)
    assert p[90] == -1 and p[15] == -1
    assert meta["end"][p == 0].max() < meta["start"][p == 1].min()
    assert meta["end"][p == 1].max() < meta["start"][p == 2].min()
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        assert not np.intersect1d(h[p == i], h[p == j]).size
    ids1, ids2 = sample_ids(p, 10, 42), sample_ids(p, 10, 42)
    for key in ids1:
        np.testing.assert_array_equal(ids1[key], ids2[key])


def test_encoded_collision_purge():
    train = np.array([[0, 1], [1, 0]], np.float32)
    val = np.array([[1, 0], [.5, .5], [0, 1]], np.float32)
    np.testing.assert_array_equal(disjoint_encoded([train], val), [False, True, False])


def test_training_only_track_a_and_reproducible_mi(frame):
    poisoned = frame.copy()
    poisoned.loc[frame.Label == 1, "IN_BYTES"] = 1e15
    a, ap = track_a(frame), track_a(poisoned)
    assert a["retained_numeric"] == ap["retained_numeric"]
    assert a["rows"] == ap["rows"]
    b, bp = track_b(frame, seed=7, max_rows=300), track_b(frame, seed=7, max_rows=300)
    assert b["ranking"] == bp["ranking"]
    assert b["mrmr_steps"] == bp["mrmr_steps"]
    assert next(x for x in b["ranking"] if x["feature"] == "PROTOCOL")["discrete"]
    assert next(x for x in b["ranking"] if x["feature"] == "MIN_TTL")["discrete"]


def test_threshold_metrics_and_collapse():
    # Deliberately tiny synthetic score set; not actual dataset/model performance.
    calibration = calibrate([1, 2, 2, 3, 4], .2)
    assert calibration["threshold"] == 4 and calibration["measured_calibration_fpr"] == 0
    m = metrics([0, 1, 3, 4], [0, 0, 1, 1], ["Benign", "Benign", "a", "b"], 2)
    assert m["roc_auc"] == 1 and m["recall"] == 1 and m["measured_fpr"] == 0
    assert m["per_attack"]["a"]["recall"] == 1
    assert embedding_diagnostics(np.zeros((20, 8)), np.ones(8))["possible_collapse"]
    with pytest.raises(ValueError):
        calibrate([np.nan])
    with pytest.raises(ValueError):
        metrics([1, 2], [0, 0], ["Benign", "Benign"], 1)


def test_end_to_end_offline_export(frame, tmp_path):
    csv = tmp_path / "fixture.csv"
    frame.to_csv(csv, index=False)
    args = Namespace(output=tmp_path / "run", data=csv, download=False,
        chunk_size=250, sample_rows=400, mi_rows=200, seed=7, train_fraction=.6,
        validation_fraction=.2, gap_ms=0, correlation=.95, modal_limit=.9995,
        mrmr_penalty=.1, target_fpr=.01)
    run(args)
    selection = json.loads((args.output / "selection.json").read_text())
    validation_ids = []
    for name, candidate in selection["candidates"].items():
        p = Preprocessor.load(args.output / name / "preprocessing.json")
        with np.load(args.output / name / "baseline.npz", allow_pickle=False) as data:
            assert data["x_train"].shape[1] == candidate["encoded_input_dimension"]
            assert (frame.loc[data["train_row_ids"], "Label"] == 0).all()
            validation_ids.append(data["validation_row_ids"])
            assert ("x_test" in data) == (name == selection["recommended_candidate"])
        raw = pd.read_csv(args.output / name / "example_raw.csv", float_precision="round_trip")
        gold = pd.read_csv(args.output / name / "example_vectors.csv").to_numpy(np.float32)
        np.testing.assert_array_equal(p.transform(raw), gold)
    assert all(np.array_equal(validation_ids[0], other) for other in validation_ids)
    assert selection["test_used_for_selection"] is False
    assert selection["validation_attack_labels_used_for_selection"] is False


def test_score_cli_binds_contract_rows_and_write_once_receipt(frame, tmp_path, monkeypatch):
    from svdd_features import evaluation
    candidate = tmp_path / "A_compact"
    candidate.mkdir()
    pipe = Preprocessor.fit(frame, ["IN_BYTES"])
    pipe.save(candidate / "preprocessing.json")
    sha = hashlib.sha256((candidate / "preprocessing.json").read_bytes()).hexdigest()
    freeze = tmp_path / "selection.json"
    freeze.write_text(json.dumps({"recommended_candidate": "A_compact", "candidates": {
        "A_compact": {"preprocessing_sha256": sha}}}))
    # Synthetic arrays solely exercise score handoff checks; no model experiment.
    y = np.array([0, 0, 1, 1], np.uint8)
    a = np.array(["Benign", "Benign", "a", "b"])
    ids = np.arange(4)
    np.savez(candidate / "baseline.npz", validation_row_ids=ids, y_validation=y,
             attack_validation=a, test_row_ids=ids+10, y_test=y, attack_test=a)
    scores = tmp_path / "scores.npz"
    payload = dict(candidate="A_compact", model_id="synthetic-only", seed=7,
                   contract_sha256=sha, validation_scores=np.array([0., 1., 3., 4.]),
                   validation_row_ids=ids, test_scores=np.array([0., 1., 3., 4.]), test_row_ids=ids+10)
    np.savez(scores, **payload)
    argv = ["evaluation", "--scores", str(scores), "--freeze", str(freeze), "--output", str(tmp_path / "scores.json"), "--final-test"]
    monkeypatch.setattr(sys, "argv", argv)
    evaluation.main()
    assert (tmp_path / "final_test_seed_7.json").exists()
    with pytest.raises(FileExistsError):
        evaluation.main()
    payload["validation_row_ids"] = ids[::-1]
    np.savez(scores, **payload)
    with pytest.raises(ValueError, match="row IDs/order mismatch"):
        evaluation.main()
