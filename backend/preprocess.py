import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path

DATA_DIR = Path("./data")
OUT_DIR = Path("./processed_unbalanced") 
OUT_DIR.mkdir(exist_ok=True)

def load_all_csv():
    files = list(DATA_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")
    print(f"Found {len(files)} CSV files.")
    dfs = [pd.read_csv(f, low_memory=False) for f in files]
    return pd.concat(dfs, ignore_index=True)

def clean_and_prepare(df):
    df.columns = df.columns.str.strip()

    base_cols = [
        'Flow ID', 'Source IP', 'Source Port', 'Destination IP',
        'Destination Port', 'Protocol', 'Timestamp'
    ]

    alt_cols = [' Src IP', ' Dst IP', ' Src Port', ' Dst Port', ' Protocol']
    cols_to_drop = [c for c in base_cols + alt_cols if c in df.columns]

    if cols_to_drop:
        print(f"Dropping metadata columns: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)

    if 'Label' not in df.columns:
        raise ValueError("No 'Label' column found. Check dataset formatting.")

    feature_cols = df.columns.drop('Label')
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(axis=1, how='all')

    before_rows = df.shape[0]
    df = df.dropna()
    after_rows = df.shape[0]
    print(f"Removed {before_rows - after_rows} dirty rows (containing NaN/Inf).")

    if df.empty:
        raise ValueError("Dataset is empty after cleaning.")

    return df

def preprocess():
    print("Loading raw data...")
    df = load_all_csv()
    print(f"Loaded: {df.shape}")

    print("Cleaning...")
    df = clean_and_prepare(df)
    print(f"After cleaning: {df.shape}")

    X = df.drop(columns=['Label'])
    y = df['Label']

    feature_names = X.columns.tolist()
    joblib.dump(feature_names, OUT_DIR / "feature_names.pkl")
    print("Saved feature_names.pkl")

    print("\nLabel distribution BEFORE encoding:")
    print(y.value_counts())

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    if X_train.empty:
        raise ValueError("X_train is empty after cleaning.")

    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    print("Skipping data balancing. Data is unbalanced.")

    print(f"Saving artifacts to {OUT_DIR}/")
    joblib.dump(X_train_scaled, OUT_DIR / "X_train.pkl")
    joblib.dump(X_test_scaled, OUT_DIR / "X_test.pkl")
    joblib.dump(y_train, OUT_DIR / "y_train.pkl")
    joblib.dump(y_test, OUT_DIR / "y_test.pkl")
    joblib.dump(scaler, OUT_DIR / "scaler.pkl")
    joblib.dump(le, OUT_DIR / "label_encoder.pkl")

    print("\nPreprocessing complete (unbalanced).")
    print(f"Training shape: {X_train_scaled.shape}")
    print(f"Test shape: {X_test_scaled.shape}")

if __name__ == "__main__":
    try:
        preprocess()
    except Exception as e:
        print(f"Error: {e}")
