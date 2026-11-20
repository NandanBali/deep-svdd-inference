"""Synthetic correctness tests; never dataset performance evidence."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
torch = pytest.importorskip("torch", reason="Install the software extra for training/reference tests")

from test_workflow import frame
from svdd_baseline.model import DeepSVDD, objective
from svdd_baseline.training import train, trace_numpy, load_model
from svdd_baseline.reference import export, verify, load_parameters, numpy_infer
from svdd_baseline.quantization import fixed_infer, round_shift
from svdd_baseline.run import execute, golden_indices
from svdd_features.preprocessing import Preprocessor
from svdd_baseline.common import sha, save_json


def policy():
    return dict(learning_rate=.001, regularization=1e-6, batch_size=128, max_epochs=6,
                min_epochs=1, patience=1, relative_min_delta=1e-4, center_epsilon=.1, cpu_threads=1)


def test_model_objective_center_and_shape():
    torch.manual_seed(7)
    model = DeepSVDD(8)
    assert sum(p.numel() for p in model.parameters()) == 896
    assert all(layer.bias is None for layer in model.layers)
    x = torch.zeros((120, 8))
    original = model.initialize_center(x)
    assert original == [0.]*8
    torch.testing.assert_close(model.center, torch.full((8,), .1))
    assert not model.center.requires_grad
    total, distance, penalty = objective(model, x, .2)
    torch.testing.assert_close(distance, torch.tensor(.08))
    torch.testing.assert_close(penalty, .1*sum(p.square().sum() for p in model.parameters()))
    torch.testing.assert_close(total, distance+penalty)
    center = model.center.clone()
    total.backward()
    torch.optim.Adam(model.parameters(), lr=.001, weight_decay=0).step()
    assert torch.equal(center, model.center)
    assert model(torch.zeros((2, 8))).shape == (2, 8)


def test_early_stop_and_checkpoint_identity(tmp_path):
    model, metadata = train(tmp_path, np.zeros((120, 8), np.float32), np.zeros((30, 8), np.float32),
                            7, policy(), "cpu", "synthetic-identity")
    assert metadata["best_epoch"] == 1 and metadata["epochs_run"] == 2
    loaded, _ = load_model(tmp_path)
    assert torch.equal(loaded.center, model.center)
    assert metadata["embedding_diagnostics"]["possible_collapse"]
    (tmp_path/"model.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="identity"):
        load_model(tmp_path)


def test_configurable_standard_preprocessing(frame, tmp_path):
    normal = frame.loc[frame.Label == 0]
    p = Preprocessor.fit(normal, ["IN_BYTES"], scaler="standard", log_mode="none", clipping=False)
    x = p.transform(normal)
    assert abs(float(x.mean())) < 1e-6 and abs(float(x.std())-1) < 1e-6
    p.save(tmp_path/"prep.json")
    np.testing.assert_array_equal(x, Preprocessor.load(tmp_path/"prep.json").transform(normal))
    poisoned = frame.copy()
    poisoned.loc[poisoned.Label == 1, "IN_BYTES"] = 1e15
    pp = Preprocessor.fit(poisoned.loc[poisoned.Label == 0], ["IN_BYTES"], scaler="standard", log_mode="none", clipping=False)
    assert p.contract == pp.contract


def test_fixed_rounding_and_widened_distance():
    np.testing.assert_array_equal(round_shift(np.array([1, 3, 5, -1, -3, -5], np.int64), 1), [0, 2, 2, 0, -2, -2])
    p = dict(W0=np.zeros((32, 8), np.float32), W1=np.zeros((16, 32), np.float32), W2=np.zeros((8, 16), np.float32), center=np.zeros(8, np.float32), threshold=np.array(.1, np.float32))
    for w in [p["W0"], p["W1"], p["W2"]]:
        for i in range(8): w[i, i] = 1
    x = np.array([[.5, .25, 1, 2, 0, 0, 0, 0]], np.float32)
    fixed, info = fixed_infer(p, x, 16, 6)
    expected = numpy_infer(p, x)
    for k in expected:
        np.testing.assert_array_equal(fixed[k], expected[k])
    assert info["score_accumulator_conservative_bits"] > 16
    _, info = fixed_infer(p, x*1000, 16, 6)
    assert info["saturation"]["inputs"]["saturated"] > 0


def test_numpy_cpp_export_and_order(frame, tmp_path):
    torch.manual_seed(9)
    model = DeepSVDD(8)
    columns = ["IN_BYTES", "OUT_BYTES", "FLOW_DURATION_MILLISECONDS", "MIN_TTL", "MAX_IP_PKT_LEN", "MIN_IP_PKT_LEN", "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT"]
    pipe = Preprocessor.fit(frame, columns)
    x = pipe.transform(frame.iloc[:120])
    model.initialize_center(torch.from_numpy(x))
    expected = trace_numpy(model, x)
    metadata = dict(checkpoint_sha256="a"*64, seed=9, architecture=[8, 32, 16, 8])
    export(tmp_path, model, pipe, .1, metadata, "synthetic")
    report = verify(tmp_path, x, expected)
    assert report["comparisons"]["cpp"]["all_nonborderline_decisions_match"]
    parameters, meta = load_parameters(tmp_path)
    parameters["W0"] = parameters["W0"].T.copy()
    np.savez(tmp_path/"parameters.npz", **parameters)
    meta["parameters_sha256"] = sha(tmp_path/"parameters.npz")
    save_json(tmp_path/"metadata.json", meta)
    with pytest.raises(ValueError, match="orientation"):
        load_parameters(tmp_path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_float_forward_matches_cpu():
    torch.manual_seed(7)
    model = DeepSVDD(8).eval()
    x = np.random.default_rng(4).normal(size=(120, 8)).astype(np.float32)
    cpu = trace_numpy(model, x)
    gpu = trace_numpy(model.cuda(), x, "cuda")
    for k in cpu:
        np.testing.assert_allclose(cpu[k], gpu[k], atol=3e-6, rtol=1e-4)


def test_full_synthetic_baseline_and_resume(frame, tmp_path):
    csv = tmp_path/"synthetic.csv"
    frame.to_csv(csv, index=False)
    c = json.loads(Path("configs/software.json").read_text())
    c.update(data=str(csv), synthetic=True, sample_rows=400, chunk_size=250, gap_ms=0,
             eligible_seeds=[7], compact_seeds=[7], golden_rows=100)
    c["training"].update(max_epochs=2, min_epochs=1, patience=1, cpu_threads=1, batch_size=128)
    c["benchmark"].update(single_warmup=1, single_calls=2, batch_warmup=1, batch_calls=2, trials=2)
    out = tmp_path/"software"
    result = execute(c, out, "cpu")
    assert result["dataset"]["experiment_data_kind"] == "synthetic smoke fixture"
    assert (out/"completion.json").exists()
    assert result["models"]["compact_seed7"]["training"]["training_rows"] == result["split"]["software_sample"]["normal_training_rows"]
    diagnostics = json.loads((out/"diagnostics.json").read_text())
    assert all(not p["test_evaluated"] and not p["baseline_training_changed"] for p in diagnostics["label_assisted_validation_probes"])
    assert diagnostics["diagnostic_device"] == "cpu"
    before = (out/"compact_seed7/test_receipt.json").read_bytes()
    second = execute(c, out, "cpu")
    assert before == (out/"compact_seed7/test_receipt.json").read_bytes()
    assert result["models"]["compact_seed7"]["test"] == second["models"]["compact_seed7"]["test"]
    changed = copy.deepcopy(c)
    changed["training"]["learning_rate"] = .01
    with pytest.raises(ValueError, match="another configuration"):
        execute(changed, out, "cpu")


def test_diagnostic_intervals_and_single_class_cohort():
    from svdd_baseline.diagnostics import interval, cohort
    lo, hi = interval(11, 11)
    assert .71 < lo < .72 and hi == 1
    assert interval(0, 0) == [None, None]
    c = cohort(np.array([0., 2.]), np.array([0, 0]), np.array(["Benign", "Benign"]), 1.)
    assert c["roc_auc"] is None and c["recall"] is None and c["measured_fpr"] == .5
