import joblib
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "processed_unbalanced"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

def train_dt():
    X_train = joblib.load(DATA_DIR / "X_train.pkl")
    X_test = joblib.load(DATA_DIR / "X_test.pkl")
    y_train = joblib.load(DATA_DIR / "y_train.pkl")
    y_test = joblib.load(DATA_DIR / "y_test.pkl")

    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        class_weight="balanced",
        random_state=42
    )

    print("Training Decision Tree")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("\nAccuracy:", accuracy_score(y_test, preds))
    print("\nClassification Report:\n")
    print(classification_report(y_test, preds, zero_division=0))

    joblib.dump(model, MODEL_DIR / "model_dt.pkl")
    print("Saved → model_dt.pkl")

if __name__ == "__main__":
    train_dt()
