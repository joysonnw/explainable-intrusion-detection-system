import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"

def _import_shap():
    import shap
    return shap


def _ensure_df_row(x: Any, feature_names: Optional[List[str]]) -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        df = x.copy()
    elif isinstance(x, (list, tuple, np.ndarray)):
        arr = np.asarray(x)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        df = pd.DataFrame(arr, columns=feature_names if feature_names else None)
    else:
        raise TypeError("Unsupported instance format for SHAP.")
    if feature_names is not None:
        missing = [f for f in feature_names if f not in df.columns]
        if missing:
            raise ValueError(f"Missing features for SHAP: {missing[:10]}...")
        df = df[feature_names]
    return df.iloc[:1]  


def _load_ensemble_and_scaler():
    ensemble = joblib.load(MODEL_DIR / "model_ensemble.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler_ensemble.pkl")
    try:
        feature_names = joblib.load(MODEL_DIR / "feature_names.pkl")
    except Exception:
        feature_names = None
    return ensemble, scaler, feature_names


def explain_instance(instance: Any,
                     method: str = "auto",
                     background: Optional[np.ndarray] = None) -> Dict[str, Any]:

    shap = _import_shap()
    ensemble, scaler, feature_names = _load_ensemble_and_scaler()

    row_df = _ensure_df_row(instance, feature_names)
    row_np = row_df.values
    row_scaled = scaler.transform(row_np)

    base_model = None
    try:
        for name, est in ensemble.estimators:
            est_name = name.lower()
            if any(k in est_name for k in ["rf", "xgb", "lgbm", "gbm", "tree"]):
                base_model = est
                break
    except Exception:
        base_model = None

    if method in ("auto", "tree") and base_model is not None:
        try:
            explainer = shap.TreeExplainer(base_model)
            shap_values = explainer.shap_values(row_scaled)
            return {
                "method": "tree",
                "model": base_model.__class__.__name__,
                "feature_names": feature_names or [f"f{i}" for i in range(row_scaled.shape[1])],
                "values": shap_values if isinstance(shap_values, list) else [shap_values],
                "expected_value": explainer.expected_value if isinstance(explainer.expected_value, list) else [explainer.expected_value],
            }
        except Exception:
            pass

    if background is None:
        background = np.zeros((1, row_scaled.shape[1]))

    def f(z):
        z = np.asarray(z)
        z = scaler.inverse_transform(z)
        z_scaled = scaler.transform(z)
        return ensemble.predict_proba(z_scaled)

    explainer = shap.KernelExplainer(f, background)
    shap_values = explainer.shap_values(row_scaled)
    return {
        "method": "kernel",
        "feature_names": feature_names or [f"f{i}" for i in range(row_scaled.shape[1])],
        "values": shap_values if isinstance(shap_values, list) else [shap_values],
        "expected_value": getattr(explainer, "expected_value", None),
    }
