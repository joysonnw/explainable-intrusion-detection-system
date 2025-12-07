import warnings
warnings.filterwarnings("ignore")

import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd 

ROOT = Path(__file__).resolve().parents[1]


DATA_DIR = ROOT / "processed_unbalanced" 
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)


def get_xgb():
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(
            eval_metric="logloss",
            n_estimators=120,       
            learning_rate=0.1,
            max_depth=6,           
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1
        )
    except Exception:
        return None


def get_lgbm():
    try:
        import lightgbm as lgb
        return lgb.LGBMClassifier(
            n_estimators=180,       
            learning_rate=0.1,
            max_depth=-1,
            num_leaves=40,           
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1
        )
    except Exception:
        return None


def train_ensemble():
    print(f"\nLoading training data from {DATA_DIR}")
    X_train = joblib.load(DATA_DIR / "X_train.pkl")
    X_test  = joblib.load(DATA_DIR / "X_test.pkl")
    y_train = joblib.load(DATA_DIR / "y_train.pkl")
    y_test  = joblib.load(DATA_DIR / "y_test.pkl")
    
    feature_names = joblib.load(DATA_DIR / "feature_names.pkl")
    le = joblib.load(DATA_DIR / "label_encoder.pkl")

    print("Training samples:", X_train.shape, " Test samples:", X_test.shape)

    joblib.dump(feature_names, MODEL_DIR / "feature_names.pkl")
    print(f"Saved feature_names.pkl to {MODEL_DIR}")


    print(f"\nLoading scaler from {DATA_DIR}")
    scaler = joblib.load(DATA_DIR / "scaler.pkl")
    X_train_s = X_train
    X_test_s  = X_test 


    dt = DecisionTreeClassifier(
        max_depth=15,
        min_samples_split=10,
        class_weight="balanced"
    )

    rf = RandomForestClassifier(
        n_estimators=120,        
        max_depth=20,
        min_samples_split=10,
        n_jobs=-1,
        class_weight="balanced"
    )

    xgb = get_xgb()
    lgbm = get_lgbm()

    estimators = [
        ("dt", dt),
        ("rf", rf)
    ]

    if xgb:
        estimators.append(("xgb", xgb))

    if lgbm:
        estimators.append(("lgbm", lgbm))


    print("\nTraining soft-voting ensemble")
    ensemble = VotingClassifier(
        estimators=estimators,
        voting="soft",
        flatten_transform=True
    )

    ensemble.fit(X_train_s, y_train)

    preds = ensemble.predict(X_test_s)
    
    class_names_str = le.classes_
    print("\nAccuracy:", accuracy_score(y_test, preds))
    print("\nClassification Report:\n")
    print(classification_report(y_test, preds, target_names=class_names_str, zero_division=0))

    print("\nSaving model artifacts")

    joblib.dump(ensemble, MODEL_DIR / "model_ensemble.pkl")
    print("Saved model_ensemble.pkl")

    joblib.dump(scaler, MODEL_DIR / "scaler_ensemble.pkl")
    print("Saved scaler_ensemble.pkl")

    joblib.dump(class_names_str.tolist(), MODEL_DIR / "class_names.pkl")
    print("Saved class_names.pkl")

    print("\nAll artifacts saved successfully!")
    print("Location:", MODEL_DIR.resolve())


if __name__ == "__main__":
    train_ensemble()