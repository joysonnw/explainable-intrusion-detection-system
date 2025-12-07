import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "processed_unbalanced"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

def train_random_forest():
    X_train = joblib.load(DATA_DIR / "X_train.pkl")
    X_test = joblib.load(DATA_DIR / "X_test.pkl")
    y_train = joblib.load(DATA_DIR / "y_train.pkl")
    y_test = joblib.load(DATA_DIR / "y_test.pkl")

    model = RandomForestClassifier(
    n_estimators=120,      
    max_depth=20,          
    min_samples_split=5,   
    n_jobs=-1,             
    class_weight="balanced",
    random_state=42
)

    print("Training Random Forest")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("\nAccuracy:", accuracy_score(y_test, preds))
    print("\nClassification Report:\n")
    print(classification_report(y_test, preds, zero_division=0))

    joblib.dump(model, MODEL_DIR / "model_rf.pkl")
    print("Saved → model_rf.pkl")

if __name__ == "__main__":
    train_random_forest()
