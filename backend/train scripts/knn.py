import joblib
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "processed_unbalanced"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

def train_knn():
    X_train = joblib.load(DATA_DIR / "X_train.pkl")
    X_test = joblib.load(DATA_DIR / "X_test.pkl")
    y_train = joblib.load(DATA_DIR / "y_train.pkl")
    y_test = joblib.load(DATA_DIR / "y_test.pkl")

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = KNeighborsClassifier(
        n_neighbors=5,
        weights="distance"
    )

    print("Training KNN")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("\nAccuracy:", accuracy_score(y_test, preds))
    print("\nClassification Report:\n")
    print(classification_report(y_test, preds, zero_division=0))

    joblib.dump(model, MODEL_DIR / "model_knn.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler_knn.pkl")
    print("Saved → model_knn.pkl & scaler_knn.pkl")

if __name__ == "__main__":
    train_knn()
