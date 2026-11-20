"""Explicit integer arithmetic experiment, NOT bit-accurate HLS verification."""
from __future__ import annotations

import math
import numpy as np
from .common import ranges


def round_shift(x, fractional_bits):
    """Signed nearest-even right shift without floating-point division."""
    x = np.asarray(x, np.int64)
    if fractional_bits == 0:
        return x.copy()
    a = np.abs(x)
    q, rem = a >> fractional_bits, a & ((1 << fractional_bits)-1)
    half = 1 << (fractional_bits-1)
    q += ((rem > half) | ((rem == half) & ((q & 1) != 0))).astype(np.int64)
    return np.where(x < 0, -q, q)


def fixed_infer(parameters, x, bits, integer_bits):
    _, raw, details = fixed_infer_codes(parameters, x, bits, integer_bits)
    scale = 1 << details["fractional_bits"]
    traces = {name: values.astype(float)/scale for name, values in raw.items()
              if name in ("h1", "h2", "embedding")}
    traces["distance_terms"] = raw["distance_terms"].astype(float)/(scale*scale)
    traces["scores"] = raw["scores"].astype(float)/(scale*scale)
    return traces, details


def fixed_infer_codes(parameters, x, bits, integer_bits):
    """Return encoded parameters and integer traces for the fixed-point contract."""
    f = bits-integer_bits
    if not 2 <= integer_bits < bits <= 24:
        raise ValueError("Simulation supports signed formats up to 24 bits")
    scale = 1 << f
    low, high = -(1 << (bits-1)), (1 << (bits-1))-1
    saturation, accumulator_ranges = {}, {}

    def clip(x, name):
        saturation[name] = dict(values=int(np.size(x)), saturated=int(np.count_nonzero((x < low) | (x > high))))
        return np.clip(x, low, high).astype(np.int64)

    def quantize(x, name):
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite quantization input")
        return clip(np.rint(np.asarray(x, np.float64)*scale), name)

    a = quantize(x, "inputs")
    center = quantize(parameters["center"], "center")
    encoded = {"center": center}
    traces = {"inputs": a}
    for i in range(3):
        key = f"W{i}"
        w = encoded[key] = quantize(parameters[key], f"weights{i}")
        width = 2*bits+math.ceil(math.log2(w.shape[1]))+1
        if width > 63:
            raise ValueError("Required accumulator exceeds int64 simulation")
        acc = a @ w.T
        accumulator_ranges[f"layer{i}"] = dict(minimum=int(acc.min()), maximum=int(acc.max()),
            conservative_required_bits=width, fractional_bits=2*f)
        a = clip(round_shift(acc, f), f"layer{i}_cast")
        if i < 2:
            a = np.maximum(a, 0)
        traces[["h1", "h2", "embedding"][i]] = a
    diff = a-center  # widening: signed bits+1, no subtraction saturation
    terms = diff*diff  # fractional_bits=2*f; retain wide products and score
    traces["distance_terms"] = terms
    traces["scores"] = terms.sum(axis=1, dtype=np.int64)
    return encoded, traces, dict(bits=bits, integer_bits_including_sign=integer_bits, fractional_bits=f,
        rounding="nearest, ties to even", overflow="saturate at input/weight/center and each layer-output cast",
        casts="Exact int64 products/sums at 2F; nearest-even shift to F; saturating B-bit cast; then ReLU on hidden layers",
        distance="Widen embedding-center to B+1 bits; keep exact square and eight-term score sum at 2F in int64; no B-bit score cast",
        score_accumulator_conservative_bits=2*(bits+1)+3,
        original_threshold="Compare real-valued integer_score / 2^(2F) to the unchanged float32 threshold; no threshold rounding in this comparison",
        accumulator_ranges=accumulator_ranges, saturation=saturation,
        claim="Arithmetic simulation only; does not verify Vitis/HLS scheduling, ap_fixed behavior or FPGA resource/throughput")


def numerical_ranges(parameters, x, trace):
    entries = {"inputs": ranges(x), **{k: ranges(v) for k, v in trace.items()},
               **{k: ranges(v) for k, v in parameters.items() if k.startswith("W") or k == "center"}}
    for entry in entries.values():
        maximum = max(entry["abs_maximum"]*1.25, 2**-16)
        integer = max(2, math.ceil(math.log2(maximum+2**-16))+1)
        entry["suggested_signed_24bit_integer_bits_including_sign"] = integer
        entry["headroom_factor"] = 1.25
    return entries
