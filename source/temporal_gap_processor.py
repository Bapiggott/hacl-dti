"""Small helper copied/adapted for DTI STGNN time-gap features."""
from __future__ import annotations
import numpy as np


def normalize_time_gaps(gaps, method: str = "log"):
    gaps = np.asarray(gaps, dtype=np.float32)
    gaps = np.nan_to_num(gaps, nan=6.0, posinf=6.0, neginf=0.0)
    gaps = np.maximum(gaps, 0.0)

    if method == "raw":
        return gaps
    if method == "log":
        return np.log1p(gaps / 12.0).astype(np.float32)
    if method == "minmax":
        mn, mx = float(gaps.min()), float(gaps.max())
        if mx <= mn:
            return np.zeros_like(gaps, dtype=np.float32)
        return ((gaps - mn) / (mx - mn)).astype(np.float32)
    if method == "buckets":
        # 0: <=6 mo, 1: <=12, 2: <=24, 3: >24, scaled to [0,1]
        bins = np.digitize(gaps, [6, 12, 24], right=True).astype(np.float32)
        return (bins / 3.0).astype(np.float32)
    raise ValueError(f"Unknown time normalization: {method}")
