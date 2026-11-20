"""Independent exported-parameter inference and portable C++ reference generation."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import numpy as np

from .common import save_json, sha


def numpy_infer(parameters, x):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != parameters["W0"].shape[1]:
        raise ValueError("Network input dimension mismatch")
    h1 = np.maximum(x @ parameters["W0"].T, np.float32(0))
    h2 = np.maximum(h1 @ parameters["W1"].T, np.float32(0))
    z = h2 @ parameters["W2"].T
    terms = (z-parameters["center"])**2
    return dict(h1=h1, h2=h2, embedding=z, distance_terms=terms,
                scores=np.sum(terms, axis=1, dtype=np.float32))


def load_parameters(folder):
    folder = Path(folder)
    metadata = json.loads((folder/"metadata.json").read_text())
    if sha(folder/"parameters.npz") != metadata["parameters_sha256"]:
        raise ValueError("Exported parameter identity mismatch")
    with np.load(folder/"parameters.npz", allow_pickle=False) as archive:
        p = {k: archive[k] for k in archive.files}
    d = metadata["architecture"]
    for i in range(3):
        if p[f"W{i}"].shape != (d[i+1], d[i]) or p[f"W{i}"].dtype != np.float32:
            raise ValueError("Wrong exported matrix orientation/type")
    if p["center"].shape != (8,) or p["threshold"].shape != ():
        raise ValueError("Wrong exported center/threshold dimensions")
    if not all(np.isfinite(a).all() for a in p.values()):
        raise ValueError("Nonfinite exported parameter")
    return p, metadata


def literal(x):
    s = format(float(x), ".9g")
    return s+(".0" if "." not in s and "e" not in s.lower() else "")+"f"


def write_cpp(folder, parameters, model_id):
    folder = Path(folder)
    lines = ["#pragma once", "// Generated float32 parameters; W[out][in], row-major.",
             f'// Model identity: {model_id}', "namespace svdd {",
             f"inline constexpr int INPUT_DIM = {parameters['W0'].shape[1]};"]
    for i in range(3):
        a = parameters[f"W{i}"]
        lines += [f"inline constexpr float W{i}[{a.shape[0]}][{a.shape[1]}] = {{"]
        lines += ["  {"+", ".join(literal(v) for v in row)+"}," for row in a]
        lines += ["};"]
    lines += ["inline constexpr float CENTER[8] = {"+", ".join(literal(v) for v in parameters["center"])+"};",
              "inline constexpr float THRESHOLD = "+literal(parameters["threshold"])+";", "}"]
    (folder/"model_parameters.h").write_text("\n".join(lines)+"\n")
    (folder/"reference.cpp").write_text(r'''#include "model_parameters.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

// Input is a headerless CSV of preprocessed float32 vectors. No preprocessing here.
int main(int argc, char** argv) {
  if (argc != 2) return 2;
  std::ifstream input(argv[1]);
  if (!input) return 3;
  std::cout << std::setprecision(9);
  std::string line;
  while (std::getline(input, line)) {
    std::istringstream row(line); std::string token;
    float x[svdd::INPUT_DIM], h1[32], h2[16], z[8], terms[8];
    for (int i=0; i<svdd::INPUT_DIM; ++i) {
      if (!std::getline(row, token, ',')) return 4;
      std::size_t used=0; x[i]=std::stof(token, &used);
      if (used != token.size()) return 5;
    }
    if (std::getline(row, token, ',')) return 6;
    for (int o=0; o<32; ++o) {
      float a=0; for (int i=0; i<svdd::INPUT_DIM; ++i) a += svdd::W0[o][i]*x[i];
      h1[o]=a>0 ? a : 0;
    }
    for (int o=0; o<16; ++o) {
      float a=0; for (int i=0; i<32; ++i) a += svdd::W1[o][i]*h1[i];
      h2[o]=a>0 ? a : 0;
    }
    float score=0;
    for (int o=0; o<8; ++o) {
      float a=0; for (int i=0; i<16; ++i) a += svdd::W2[o][i]*h2[i];
      z[o]=a; float d=a-svdd::CENTER[o]; terms[o]=d*d; score += terms[o];
    }
    for (float v:h1) std::cout << v << ',';
    for (float v:h2) std::cout << v << ',';
    for (float v:z) std::cout << v << ',';
    for (float v:terms) std::cout << v << ',';
    std::cout << score << ',' << (score>svdd::THRESHOLD ? 1 : 0) << '\n';
  }
}
''')


def export(folder, model, pipe, threshold, training_metadata, experiment_id):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    p = {f"W{i}": layer.weight.detach().cpu().numpy().copy() for i, layer in enumerate(model.layers)}
    p.update(center=model.center.detach().cpu().numpy().copy(), threshold=np.array(threshold, np.float32))
    np.savez(folder/"parameters.npz", **p)
    for k, a in p.items():
        np.save(folder/f"{k}.npy", a, allow_pickle=False)
    pipe.save(folder/"preprocessing.json")
    metadata = dict(experiment_id=experiment_id, checkpoint_sha256=training_metadata["checkpoint_sha256"],
        seed=training_metadata["seed"], architecture=training_metadata["architecture"],
        model_id=f"{experiment_id}:{training_metadata['seed']}:{training_metadata['checkpoint_sha256']}",
        ordered_raw_features=pipe.contract["raw_columns"], ordered_pl_inputs=pipe.contract["output_names"],
        operations=["PS: documented preprocessing", "PL: W0*x", "ReLU", "W1*h1", "ReLU", "W2*h2", "sum((embedding-center)^2)", "score > threshold"],
        matrix_orientation="W[out_features,in_features], row-major; y=W*x; NumPy batches use x @ W.T",
        numerical_types="float32 inputs, weights, center, activations, products, distance terms, score and threshold; decision bool",
        threshold=float(p["threshold"]), center=p["center"].tolist(),
        fpga_scope="Already-preprocessed vector enters PL; software preprocessing initially runs on PS. No packet parsing, training or online adaptation.",
        parameters_sha256=sha(folder/"parameters.npz"), preprocessing_sha256=sha(folder/"preprocessing.json"))
    save_json(folder/"metadata.json", metadata)
    write_cpp(folder, p, metadata["model_id"])
    return p, metadata


def verify(folder, x, expected, row_ids=None):
    folder = Path(folder)
    p, metadata = load_parameters(folder)
    np.savez_compressed(folder/"golden_expected.npz", inputs=x, **expected,
                        decisions=expected["scores"] > p["threshold"],
                        row_ids=np.arange(len(x)) if row_ids is None else row_ids)
    np.savetxt(folder/"golden_inputs.csv", x, delimiter=",", fmt="%.9g")
    np.savetxt(folder/"golden_outputs.csv", np.column_stack([expected[k] for k in ["h1", "h2", "embedding", "distance_terms"]]+[expected["scores"], expected["scores"] > p["threshold"]]), delimiter=",", fmt="%.9g")
    compiler = shutil.which("g++")
    if not compiler:
        raise RuntimeError("C++ verification requires g++ (Vivado is not required)")
    command = [compiler, "-std=c++17", "-O2", "-ffp-contract=off", "-fno-fast-math",
               str(folder/"reference.cpp"), "-o", str(folder/"reference")]
    subprocess.run(command, check=True, capture_output=True, text=True)
    result = subprocess.run([str((folder/"reference").resolve()), str((folder/"golden_inputs.csv").resolve())], check=True, capture_output=True, text=True)
    from io import StringIO
    cpp = np.loadtxt(StringIO(result.stdout), delimiter=",", dtype=np.float32, ndmin=2)
    if cpp.shape != (len(x), 66):
        raise ValueError(f"C++ trace dimensions are wrong: {cpp.shape}")
    cpp_trace, offset = {}, 0
    for k, width in [("h1", 32), ("h2", 16), ("embedding", 8), ("distance_terms", 8)]:
        cpp_trace[k] = cpp[:, offset:offset+width]
        offset += width
    cpp_trace["scores"] = cpp[:, 64]
    np.savetxt(folder/"golden_cpp_outputs.csv", cpp, delimiter=",", fmt="%.9g")
    report = dict(model_id=metadata["model_id"], rows=len(x), compiler_command=command,
        compiler_version=subprocess.check_output([compiler, "--version"], text=True).splitlines()[0],
        tolerance=dict(activation_atol=3e-6, activation_rtol=1e-4, distance_term_atol=1e-8, score_atol=1e-8, score_rtol=1e-4),
        comparisons={})
    for name, actual in [("numpy", numpy_infer(p, x)), ("cpp", cpp_trace)]:
        errors = {}
        for k, target in expected.items():
            # Squaring a subtraction near the center magnifies relative error;
            # embeddings are independently checked before these derived terms.
            atol = 1e-8 if k in {"scores", "distance_terms"} else 3e-6
            np.testing.assert_allclose(actual[k], target, rtol=1e-4, atol=atol, err_msg=f"{name} {k}")
            errors[k] = float(np.max(np.abs(actual[k].astype(float)-target)))
        decisions = actual["scores"] > p["threshold"]
        mismatch = decisions != (expected["scores"] > p["threshold"])
        borderline = np.abs(expected["scores"]-p["threshold"]) <= (1e-8+1e-4*np.abs(expected["scores"]))
        if np.any(mismatch & ~borderline):
            raise AssertionError(f"{name} decision differs outside declared numerical tolerance")
        if name == "cpp":
            np.testing.assert_array_equal(cpp[:, 65].astype(bool), decisions)
        report["comparisons"][name] = dict(max_absolute_errors=errors,
            decision_disagreements=int(mismatch.sum()), borderline_cases=int(borderline.sum()),
            disagreeing_row_ids=(np.arange(len(x)) if row_ids is None else row_ids)[mismatch].tolist(),
            all_nonborderline_decisions_match=True)
    save_json(folder/"reference_verification.json", report)
    return report
