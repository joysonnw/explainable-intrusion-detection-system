# ensemble_inference.py
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from utils import (
    to_dataframe,
    align_features,
    compute_distribution,
    ensure_2d_array,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"

class EnsemblePredictor:

    def __init__(self, model_path: Optional[Path] = None, scaler_path: Optional[Path] = None):
        self.model_path = model_path or (MODEL_DIR / "model_ensemble.pkl")
        self.scaler_path = scaler_path or (MODEL_DIR / "scaler_ensemble.pkl")

        try:
            self.ensemble = joblib.load(self.model_path)
        except Exception as e:
            raise RuntimeError(f"Could not load ensemble model from {self.model_path}: {e}")

        try:
            self.scaler = joblib.load(self.scaler_path)
        except Exception as e:
            raise RuntimeError(f"Could not load scaler from {self.scaler_path}: {e}")

        self.feature_names: Optional[List[str]] = None
        self.class_names: Optional[List[str]] = None

        try:
            self.feature_names = joblib.load(MODEL_DIR / "feature_names.pkl")
        except Exception:
            self.feature_names = None

        try:
            self.class_names = joblib.load(MODEL_DIR / "class_names.pkl")
        except Exception:
            self.class_names = None

    def _prepare(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        X = to_dataframe(X)
        if self.feature_names:
            X = align_features(X, self.feature_names)
        X_np = ensure_2d_array(X.values)
        X_scaled = self.scaler.transform(X_np)
        return X, X_scaled

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        _, Xs = self._prepare(X)
        return self.ensemble.predict(Xs)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        _, Xs = self._prepare(X)
        return self.ensemble.predict_proba(Xs)

    def infer(self, X: pd.DataFrame, return_proba: bool = True) -> Dict[str, Any]:
        df, Xs = self._prepare(X)
        preds = self.ensemble.predict(Xs)

        out: Dict[str, Any] = {"predictions": preds.tolist()}

        if return_proba:
            try:
                probs = self.ensemble.predict_proba(Xs)
                out["probabilities"] = probs.tolist()
            except Exception:
                out["probabilities"] = None

        dist = compute_distribution(preds, class_names=self.class_names)
        out["distribution"] = dist

        out["n_rows"] = int(df.shape[0])
        out["n_features"] = int(df.shape[1])
        out["feature_names"] = self.feature_names or list(df.columns)

        if not self.class_names:
            unique = sorted(np.unique(preds))
            out["class_names"] = [str(c) for c in unique]  
        else:
            out["class_names"] = self.class_names

        return out


def load_models() -> EnsemblePredictor:
    return EnsemblePredictor()


def run_inference(csv_or_df: Any) -> Dict[str, Any]:
   
    predictor = load_models()
    df = to_dataframe(csv_or_df)
    return predictor.infer(df, return_proba=True)
