"""Export the frozen primary model's fixed-point constants and integer test vectors."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .common import save_json, sha
from .quantization import fixed_infer_codes
from .reference import load_parameters


def write_header(path, name, model_id, encoded, arithmetic, thresholds):
    bits = arithmetic["bits"]
    storage = "std::int16_t" if bits <= 16 else "std::int32_t"
    lines = [
        "#pragma once",
        "#include <cstdint>",
        "",
        "// Raw fixed-point integer codes. W[out][in], row-major.",
        f"// Model identity: {model_id}",
        "namespace svdd {",
        f"namespace {name.lower()} {{",
        "",
        f"static constexpr int BITS = {bits};",
        f"static constexpr int INTEGER_BITS = {arithmetic['integer_bits_including_sign']};",
        f"static constexpr int FRACTIONAL_BITS = {arithmetic['fractional_bits']};",
        "static constexpr int SCORE_FRACTIONAL_BITS = 2 * FRACTIONAL_BITS;",
        f"static constexpr int INPUT_DIM = {encoded['W0'].shape[1]};",
    ]
    for i in range(3):
        width = arithmetic["accumulator_ranges"][f"layer{i}"]["conservative_required_bits"]
        lines.append(f"static constexpr int W{i}_ACC_BITS = {width};")
    lines += [
        f"static constexpr int SCORE_ACC_BITS = {arithmetic['score_accumulator_conservative_bits']};",
        "// Anomaly iff score_code > threshold_code.",
        f"static constexpr std::int64_t THRESHOLD = {thresholds['original']['code']};",
        f"static constexpr std::int64_t VALIDATION_THRESHOLD = {thresholds['recalibrated']['code']};",
        "",
    ]
    for i in range(3):
        w = encoded[f"W{i}"]
        lines.append(f"static constexpr {storage} W{i}[{w.shape[0]}][{w.shape[1]}] = {{")
        for row in w:
            lines.append("    {" + ", ".join(map(str, row)) + "},")
        lines += ["};", ""]
    center = ", ".join(map(str, encoded["center"]))
    lines += [
        f"static constexpr {storage} CENTER[{encoded['center'].size}] = {{{center}}};",
        "",
        f"}}  // namespace {name.lower()}",
        "}  // namespace svdd",
    ]
    path.write_text("\n".join(lines) + "\n")


def export_run(run):
    """Generate hardware files from an existing run without retraining or recalibrating."""
    freeze_path = run/"freeze.json"
    freeze = json.loads(freeze_path.read_text())
    primary = freeze["primary_model"]
    frozen_model = freeze["models"][primary]
    source = run/primary/"export"
    parameters, metadata = load_parameters(source)
    if (metadata["experiment_id"] != freeze["experiment_id"]
            or metadata["checkpoint_sha256"] != frozen_model["checkpoint_sha256"]
            or sha(source.parent/"model.pt") != frozen_model["checkpoint_sha256"]
            or float(parameters["threshold"]) != frozen_model["threshold"]
            or parameters["center"].tolist() != frozen_model["center"]
            or sha(source/"preprocessing.json") != metadata["preprocessing_sha256"]):
        raise ValueError("Float export does not match the frozen primary model")
    golden = source/"golden_expected.npz"
    with np.load(golden, allow_pickle=False) as archive:
        inputs = archive["inputs"]
    outputs = []
    for fmt in freeze["fixed_point_formats"]:
        name = f"Q{fmt['bits']}_I{fmt['integer_bits']}"
        encoded, raw, arithmetic = fixed_infer_codes(parameters, inputs, **fmt)
        score_scale = 1 << (2 * arithmetic["fractional_bits"])
        original = float(parameters["threshold"])
        calibration = freeze["quantized_validation_calibrations"][name]
        calibrated = calibration["threshold"]
        # Integer S > floor(t * 2^(2F)) is equivalent to S / 2^(2F) > t.
        thresholds = {
            "original": dict(value=original, code=math.floor(original * score_scale)),
            "recalibrated": dict(**calibration, code=math.floor(calibrated * score_scale)),
        }
        decisions = raw["scores"] > thresholds["original"]["code"]
        calibrated_decisions = raw["scores"] > thresholds["recalibrated"]["code"]
        np.testing.assert_array_equal(decisions, raw["scores"] / score_scale > original)
        np.testing.assert_array_equal(calibrated_decisions, raw["scores"] / score_scale > calibrated)

        folder = source/"fixed_point"/name
        folder.mkdir(parents=True, exist_ok=True)
        write_header(folder/"model_parameters.h", name, metadata["model_id"], encoded, arithmetic, thresholds)
        output_keys = ("h1", "h2", "embedding", "distance_terms", "scores")
        trace = np.column_stack([raw[key] for key in output_keys] + [decisions, calibrated_decisions])
        np.savetxt(folder/"golden_inputs.csv", raw["inputs"], fmt="%d", delimiter=",")
        np.savetxt(folder/"golden_outputs.csv", trace, fmt="%d", delimiter=",")
        save_json(folder/"metadata.json", dict(
            model_id=metadata["model_id"], architecture=metadata["architecture"],
            ordered_pl_inputs=metadata["ordered_pl_inputs"], matrix_orientation=metadata["matrix_orientation"],
            checkpoint_sha256=metadata["checkpoint_sha256"], preprocessing_sha256=metadata["preprocessing_sha256"],
            float_parameters_sha256=metadata["parameters_sha256"], freeze_sha256=sha(freeze_path),
            float_golden_sha256=sha(golden), golden_rows=len(inputs),
            golden_output_columns=[
                dict(name=key, count=raw[key].shape[1] if raw[key].ndim == 2 else 1) for key in output_keys
            ] + [dict(name="original_decision", count=1), dict(name="recalibrated_decision", count=1)],
            format=name, arithmetic=arithmetic, thresholds=thresholds,
        ))
        outputs.append(folder)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/software"), help="Completed baseline run directory")
    args = parser.parse_args()
    for folder in export_run(args.run):
        print(folder)


if __name__ == "__main__":
    main()
