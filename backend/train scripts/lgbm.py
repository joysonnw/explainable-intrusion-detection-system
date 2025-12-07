import joblib
from pathlib import Path
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, accuracy_score

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "processed_unbalanced"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

def train_lgbm():
    X_train = joblib.load(DATA_DIR / "X_train.pkl")
    X_test = joblib.load(DATA_DIR / "X_test.pkl")
    y_train = joblib.load(DATA_DIR / "y_train.pkl")
    y_test = joblib.load(DATA_DIR / "y_test.pkl")

    model = LGBMClassifier(
        objective="multiclass",
        num_class=len(set(y_train)),
        n_estimators=300,
        learning_rate=0.05,
        max_depth=-1,
        class_weight='balanced',
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    print("Training LightGBM")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("\nAccuracy:", accuracy_score(y_test, preds))
    print("\nClassification Report:\n")
    print(classification_report(y_test, preds, zero_division=0))

    joblib.dump(model, MODEL_DIR / "model_lgbm.pkl")
    print("Saved → model_lgbm.pkl")

if __name__ == "__main__":
    train_lgbm()
