import joblib
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "processed_unbalanced"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

def train_xgb():
    X_train = joblib.load(DATA_DIR / "X_train.pkl")
    X_test = joblib.load(DATA_DIR / "X_test.pkl")
    y_train = joblib.load(DATA_DIR / "y_train.pkl")
    y_test = joblib.load(DATA_DIR / "y_test.pkl")

    model = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        learning_rate=0.1,
        max_depth=8,
        n_estimators=200,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=42
    )

    print("Training XGBoost")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("\nAccuracy:", accuracy_score(y_test, preds))
    print("\nClassification Report:\n")
    print(classification_report(y_test, preds, zero_division=0))

    joblib.dump(model, MODEL_DIR / "model_xgb.pkl")
    print("Saved → model_xgb.pkl")

if __name__ == "__main__":
    train_xgb()
