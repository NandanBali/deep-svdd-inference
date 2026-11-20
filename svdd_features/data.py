"""Full-file chunked inspection, chronological split, bounded uniform row sampling."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import candidates, expected_columns, feature_info
from .preprocessing import valid_values


def chunks(path: Path, size=100_000):
    header = list(pd.read_csv(path, nrows=0).columns)
    if set(header) != expected_columns() or len(header) != 55:
        raise ValueError("CSV is not the audited 53-feature + 2-label v3 schema; inspect and extend audit explicitly")
    dtype = {c: str if c in {"Attack", "IPV4_SRC_ADDR", "IPV4_DST_ADDR"} else np.float64 for c in header}
    return pd.read_csv(path, dtype=dtype, chunksize=size)


def row_hash(frame: pd.DataFrame) -> np.ndarray:
    # Fixed dtypes and column order are mandatory for reproducible pandas hashes.
    return pd.util.hash_pandas_object(frame, index=False).to_numpy(np.uint64)


def inspect(path: Path, size: int = 100_000):
    stats, arrays, distribution, family_distribution, dtypes = {}, [], Counter(), Counter(), {}
    families, offset = {}, 0
    for block in chunks(path, size):
        if set(block.Label.unique()) - {0.0, 1.0}:
            raise ValueError("Unexpected binary labels")
        if block.Attack.isna().any() or ((block.Label == 0) != (block.Attack == "Benign")).any():
            raise ValueError("Binary labels disagree with Benign attack category")
        for family in block.Attack.unique():
            if family not in families:
                families[family] = len(families)
        distribution.update(block.Label.astype(int).tolist())
        family_distribution.update(block.Attack.tolist())
        candidate_frame = pd.DataFrame({c: valid_values(block, c) for c in candidates(list(block))})
        arrays.append(dict(
            raw_hash=row_hash(block), feature_hash=row_hash(candidate_frame),
            start=block.FLOW_START_MILLISECONDS.to_numpy(), end=block.FLOW_END_MILLISECONDS.to_numpy(),
            label=block.Label.to_numpy(np.uint8), family=block.Attack.map(families).to_numpy(np.uint8)))
        for c in block:
            s = stats.setdefault(c, dict(missing=0, infinite=0, negative=0, invalid=0, zeros=0,
                                        fractional=0, minimum=None, maximum=None))
            dtypes[c] = str(block[c].dtype)
            s["missing"] += int(block[c].isna().sum())
            if c in {"Attack", "IPV4_SRC_ADDR", "IPV4_DST_ADDR"}:
                continue
            x = block[c].to_numpy()
            finite = x[np.isfinite(x)]
            s["infinite"] += int(np.isinf(x).sum())
            s["negative"] += int((x < 0).sum())
            s["zeros"] += int((x == 0).sum())
            s["fractional"] += int((finite != np.floor(finite)).sum())
            if feature_info(c)["candidate"]:
                s["invalid"] += int(np.isnan(valid_values(block, c)).sum())
            if len(finite):
                lo, hi = float(finite.min()), float(finite.max())
                s["minimum"] = lo if s["minimum"] is None else min(lo, s["minimum"])
                s["maximum"] = hi if s["maximum"] is None else max(hi, s["maximum"])
        offset += len(block)
        print(f"Inspected {offset:,} flows", flush=True)
    if not offset:
        raise ValueError("Empty dataset")
    meta = {k: np.concatenate([a[k] for a in arrays]) for k in arrays[0]}
    del arrays
    for c, s in stats.items():
        s["constant_finite"] = s["minimum"] is not None and s["minimum"] == s["maximum"]
        s["parser_dtype"] = dtypes[c]
    order = np.argsort(meta["feature_hash"], kind="stable")
    h, y = meta["feature_hash"][order], meta["label"][order]
    groups = np.r_[0, np.flatnonzero(h[1:] != h[:-1]) + 1]
    conflicting = int((np.minimum.reduceat(y, groups) != np.maximum.reduceat(y, groups)).sum())
    summary = dict(
        rows=offset, columns=list(block.columns), raw_feature_count=53,
        class_distribution={str(k): int(v) for k, v in sorted(distribution.items())},
        attack_distribution=dict(sorted(family_distribution.items())),
        column_quality=stats,
        exact_duplicate_rows=int(offset-len(np.unique(meta["raw_hash"]))),
        duplicate_candidate_vectors=int(offset-len(groups)),
        candidate_vectors_with_conflicting_binary_labels=conflicting,
        hash_method="pandas 64-bit row hash with pinned version/dtypes/order; negligible nonzero collision risk, not cryptographic proof",
        family_codes={str(v): k for k, v in families.items()},
    )
    return meta, summary


def chronological_split(meta: dict, train_fraction=.6, validation_fraction=.2, gap_ms=60_000):
    if not 0 < train_fraction < train_fraction + validation_fraction < 1 or gap_ms < 0:
        raise ValueError("Invalid split fractions or embargo")
    start, end = meta["start"], meta["end"]
    valid = np.isfinite(start) & np.isfinite(end) & (end >= start) & (start > 0)
    if not valid.any():
        raise ValueError("No valid temporal metadata; an explicit documented fallback is required")
    t1, t2 = np.quantile(start[valid], [train_fraction, train_fraction + validation_fraction], method="nearest")
    if t2 <= t1:
        raise ValueError("Time cutoffs coincide; temporal split impossible")
    p = np.full(len(start), -1, dtype=np.int8)
    p[valid & (end < t1-gap_ms)] = 0
    p[valid & (start >= t1+gap_ms) & (end < t2-gap_ms)] = 1
    p[valid & (start >= t2+gap_ms)] = 2
    before = [(p == i).sum() for i in range(3)]
    # Preserve chronology: keep a feature-vector group in its earliest eligible partition;
    # purge its occurrences from later partitions. Within-partition multiplicity is preserved.
    h = meta["feature_hash"]
    _, inverse = np.unique(h, return_inverse=True)
    first = np.full(inverse.max()+1, 3, dtype=np.int8)
    np.minimum.at(first, inverse[p >= 0], p[p >= 0])
    purged_duplicate = (p >= 0) & (p != first[inverse])
    p[purged_duplicate] = -1
    if any(not np.any(p == i) for i in range(3)):
        raise ValueError("Empty partition after time/duplicate purges")
    hashes = [np.unique(h[p == i]) for i in range(3)]
    assert not any(np.intersect1d(hashes[a], hashes[b]).size for a, b in [(0, 1), (0, 2), (1, 2)])
    info = dict(
        method="chronological flow-start quantiles with whole-flow end purge and symmetric embargo",
        train_fraction=train_fraction, validation_fraction=validation_fraction, embargo_ms=gap_ms,
        cutoffs_ms=[int(t1), int(t2)],
        invalid_time_rows=int((~valid).sum()),
        time_or_boundary_purged_rows=int((p == -1).sum()-purged_duplicate.sum()),
        cross_partition_duplicate_rows_purged=int(purged_duplicate.sum()),
        counts_before_duplicate_purge=[int(x) for x in before],
        limitations="Same hosts/capture may span partitions. No independent PCAP/session IDs. Chronology reduces leakage but is not a held-out network or capture evaluation; attack-family proportions may shift.",
        partitions={},
    )
    family_names = meta.get("family_names", {})
    for i, name in enumerate(["train", "validation", "test"]):
        mask = p == i
        info["partitions"][name] = dict(
            rows=int(mask.sum()), start_min_ms=int(start[mask].min()), end_max_ms=int(end[mask].max()),
            class_distribution={str(k): int(v) for k, v in zip(*np.unique(meta["label"][mask], return_counts=True))},
            attack_distribution={family_names.get(str(k), str(k)): int(v) for k, v in zip(*np.unique(meta["family"][mask], return_counts=True))},
        )
    return p, info


def sample_ids(partitions, limit: int, seed: int):
    if limit <= 0:
        raise ValueError("Sample limit must be positive")
    result = {}
    for i, name in enumerate(["train", "validation", "test"]):
        ids = np.flatnonzero(partitions == i)
        if len(ids) > limit:
            ids = np.random.default_rng(seed+i).choice(ids, limit, replace=False)
        result[name] = np.sort(ids)
    return result


def load_ids(path: Path, ids: dict, size=100_000):
    """Read selected raw rows only; no feature fitting here."""
    blocks, offset = {k: [] for k in ids}, 0
    for block in chunks(path, size):
        for name, wanted in ids.items():
            local = wanted[(wanted >= offset) & (wanted < offset+len(block))]-offset
            if len(local):
                selected = block.iloc[local].copy()
                selected.index = wanted[(wanted >= offset) & (wanted < offset+len(block))]
                blocks[name].append(selected)
        offset += len(block)
    return {k: pd.concat(v) if v else pd.DataFrame() for k, v in blocks.items()}


def disjoint_encoded(earlier: list[np.ndarray], later: np.ndarray) -> np.ndarray:
    """Mask collisions introduced by feature removal, quantile clipping, or encoding.

    Full float32 rows are compared exactly (no hash collision risk). Do not refit
    or reselect based on the resulting validation/test retention counts.
    """
    def keys(x):
        x = np.ascontiguousarray(x, dtype=np.float32)
        return x.view(np.dtype((np.void, x.dtype.itemsize*x.shape[1]))).ravel()
    prev = np.concatenate([keys(x) for x in earlier])
    return ~np.isin(keys(later), np.unique(prev))
