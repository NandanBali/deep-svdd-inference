from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False)+"\n")
    temporary.replace(path)


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def identity(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, allow_nan=False).encode()).hexdigest()


def array_sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def ranges(x):
    x = np.asarray(x)
    return {"minimum": float(x.min()), "maximum": float(x.max()),
            "abs_maximum": float(np.abs(x).max()),
            "per_dimension_minimum": np.min(x, axis=0).tolist(),
            "per_dimension_maximum": np.max(x, axis=0).tolist()}
