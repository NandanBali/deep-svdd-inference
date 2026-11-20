"""Portable JSON preprocessing contract. Only ordered, audited features are accepted."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import feature_info


def valid_values(frame: pd.DataFrame, column: str) -> np.ndarray:
    """Coerce malformed, nonfinite, negative and out-of-domain values to missing."""
    info = feature_info(column)
    x = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64, copy=True)
    invalid = ~np.isfinite(x) | (x < 0)
    if info.get("max_value") is not None:
        invalid |= x > info["max_value"]
    if info["kind"] in {"categorical", "port", "flags"} and column != "L7_PROTO":
        invalid |= x != np.floor(x)
    x[invalid] = np.nan
    return x


def tokens(frame: pd.DataFrame, column: str) -> np.ndarray:
    x = valid_values(frame, column)
    if feature_info(column)["kind"] == "port":
        # Explicit engineering choice: four range categories, never ordinal port distance.
        return np.where(np.isnan(x), "missing", np.where(x == 0, "zero", np.where(
            x <= 1023, "system", np.where(x <= 49151, "registered", "dynamic"))))
    return np.array(["missing" if np.isnan(v) else format(v, ".12g") for v in x], dtype=object)


class Preprocessor:
    def __init__(self, contract: dict):
        self.contract = contract

    @classmethod
    def fit(cls, normal_training: pd.DataFrame, columns: list[str], clip_quantile: float = .999,
            *, scaler="quantile_anchor", log_mode="audited", output_clip=(-1., 4.), clipping=True):
        if not len(normal_training) or not 0.5 < clip_quantile <= 1:
            raise ValueError("Need normal training samples and clip quantile in (0.5,1]")
        if len(columns) != len(set(columns)):
            raise ValueError("Duplicate raw columns")
        if scaler not in {"quantile_anchor", "standard"} or log_mode not in {"audited", "none"}:
            raise ValueError("Unknown scaler/log policy")
        if len(output_clip) != 2 or not output_clip[0] < output_clip[1]:
            raise ValueError("Invalid output clipping range")
        entries, names = [], []
        for column in columns:
            info = feature_info(column)
            if not info["candidate"]:
                raise ValueError(f"Forbidden model input: {column}")
            kind = info["kind"]
            entry = {"raw_column": column, "kind": kind, "offset": len(names)}
            if kind == "flags":
                entry.update(encoding="eight_bits_lsb_first", missing_value=0,
                             output_names=[f"{column}.bit{i}" for i in range(8)])
            elif kind in {"categorical", "port"}:
                values = tokens(normal_training, column)
                if kind == "port":
                    categories = [v for v in ["zero", "system", "registered", "dynamic"] if np.any(values == v)]
                else:
                    counts = pd.Series(values).value_counts()
                    categories = sorted([v for v in counts.index if v != "missing"],
                                        key=lambda v: (-int(counts[v]), v))[:3]
                # Add an other channel only when it varies on normal training.
                # Otherwise unseen categories have an explicit all-zero encoding.
                # This avoids padding a constant reserved input into the MLP.
                other_count = int((~np.isin(values, categories)).sum())
                has_other = .0005 <= other_count/len(values) <= .9995
                entry.update(encoding="one_hot", categories=categories, has_other=has_other,
                             unknown_and_missing="other" if has_other else "all_zero",
                             other_channel_min_frequency=.0005,
                             output_names=[f"{column}={v}" for v in categories] + ([f"{column}=other"] if has_other else []))
            else:
                values = valid_values(normal_training, column)
                finite = values[np.isfinite(values)]
                median = float(np.median(finite)) if len(finite) else 0.0
                values = np.where(np.isfinite(values), values, median)
                # Preserve TTL, lengths and window geometry; compress heavy-tailed counters/times/rates.
                log = info["log1p"] and log_mode == "audited"
                if log:
                    values = np.log1p(values)
                if scaler == "standard":
                    low = float(values.mean())
                    high = low + float(values.std())
                else:
                    low, high = np.quantile(values, [1 - clip_quantile, clip_quantile])
                scale = float(high-low) if high > low else 1.0
                # A normal-only upper quantile can be 31 for IP TTL. Hard clipping
                # there would make an unseen TTL=252 identical to TTL=31. Allow a
                # bounded extension of the training scale before saturation.
                entry.update(encoding="numeric", median=median, log1p=log,
                             anchor_lower=float(low), anchor_upper=float(high),
                             clip_lower=float(low+output_clip[0]*scale) if clipping else None,
                             clip_upper=float(low+output_clip[1]*scale) if clipping else None,
                             scale=scale,
                             output_names=[column])
            names.extend(entry["output_names"])
            entries.append(entry)
        return cls({
            "version": 1, "raw_columns": columns, "entries": entries,
            "output_names": names, "input_dimension": len(names), "dtype": "float32",
            "fit_population": "normal training only", "fit_rows": len(normal_training),
            "clip_quantile": clip_quantile, "scaler": scaler, "log_mode": log_mode,
            "clipping_enabled": clipping,
            "numeric_formula": "x=median if invalid; z=log1p(x) if configured else x; (clip(z,clip_lower,clip_upper)-anchor_lower)/scale",
            "numeric_output_range": list(output_clip) if clipping else None,
            "clipping_policy": f"Normal-training {'mean/std' if scaler == 'standard' else 'quantile'} anchors; output saturates at {list(output_clip) if clipping else 'no bounds'}.",
            "invalid_behavior": "Absent required column: error. Malformed/nonfinite/negative/out-of-domain: missing. Numeric: training median; category: per-entry other or all-zero; flags: zero mask.",
            "out_of_range": f"Numeric clipping {'at saved bounds, output '+str(list(output_clip)) if clipping else 'disabled'}; beyond-normal contrast follows the saved anchors. Unknown categories follow per-entry other/all-zero rule. No refitting.",
            "extra_columns": "Ignored; never used to infer order or inputs",
            "flow_direction": "nProbe exported orientation; IN=src->dst, OUT=dst->src; do not reorder endpoints",
        })

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        missing = set(self.contract["raw_columns"]) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing required raw columns: {sorted(missing)}")
        blocks = []
        for e in self.contract["entries"]:
            c = e["raw_column"]
            if e["encoding"] == "one_hot":
                values = tokens(frame, c)
                block = np.zeros((len(frame), len(e["output_names"])), dtype=np.float32)
                for i, category in enumerate(e["categories"]):
                    block[:, i] = values == category
                if e["has_other"]:
                    block[:, -1] = ~np.isin(values, e["categories"])
            elif e["encoding"] == "eight_bits_lsb_first":
                values = np.nan_to_num(valid_values(frame, c), nan=0).astype(np.uint16)
                block = ((values[:, None] >> np.arange(8)) & 1).astype(np.float32)
            else:
                values = valid_values(frame, c)
                values = np.where(np.isfinite(values), values, e["median"])
                if e["log1p"]:
                    values = np.log1p(values)
                block = ((np.clip(values, e["clip_lower"], e["clip_upper"])-e["anchor_lower"])/e["scale"])[:, None]
            blocks.append(block)
        return np.ascontiguousarray(np.concatenate(blocks, axis=1), dtype=np.float32)

    def save(self, path: Path):
        path.write_text(json.dumps(self.contract, indent=2, allow_nan=False) + "\n")

    @classmethod
    def load(cls, path: Path):
        contract = json.loads(Path(path).read_text())
        if contract.get("version") != 1:
            raise ValueError("Unsupported preprocessing contract version")
        columns = contract["raw_columns"]
        if len(columns) != len(set(columns)) or any(not feature_info(c)["candidate"] for c in columns):
            raise ValueError("Invalid/forbidden raw columns in contract")
        names = [n for e in contract["entries"] for n in e["output_names"]]
        if names != contract["output_names"] or len(names) != contract["input_dimension"]:
            raise ValueError("Inconsistent contract dimensions/order")
        if [e["raw_column"] for e in contract["entries"]] != columns:
            raise ValueError("Inconsistent raw column order")
        return cls(contract)
