import numpy as np
import pandas as pd
from typing import Any, List, Optional, Dict

def to_dataframe(x: Any) -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        return x.copy()
    if isinstance(x, (list, tuple, np.ndarray)):
        arr = np.asarray(x)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return pd.DataFrame(arr)
    raise TypeError("Input must be a pandas DataFrame, list/tuple, or numpy array.")

def align_features(df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing[:10]}...")
    return df[feature_names]

def ensure_2d_array(x):
    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return x

def compute_distribution(preds: np.ndarray, class_names: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
    unique, counts = np.unique(preds, return_counts=True)
    total = int(counts.sum())
    out = {}
    for cls, cnt in zip(unique, counts):
        key = class_names[cls] if class_names and 0 <= cls < len(class_names) else str(int(cls))
        out[key] = {"count": int(cnt), "percent": round(float(cnt) * 100.0 / total, 4)}
    return out
