# app.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from io import BytesIO
import json
import numpy as np
from pathlib import Path
import datetime
from typing import Dict, Any
import sys, traceback
from report_generator import generate_attack_report
from fastapi.responses import FileResponse

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        return super().default(obj)

use_dummy = False
try:
    from ensemble_inference import load_models
    from xai import explain_instance
    from insights import get_actionable_insight
except Exception as e:
    print(f"WARNING: Could not import real backend modules: {e}")
    traceback.print_exc()
    use_dummy = True

class DummyPredictor:
    def __init__(self):
        self.class_names = ["BENIGN", "DDoS", "PortScan"]
        self.feature_names = ["f1", "f2", "f3"]

    def infer(self, df, return_proba=False):
        n = len(df)
        preds = np.random.choice([0, 1, 2], size=n)
        unique, counts = np.unique(preds, return_counts=True)
        total = int(counts.sum())
        dist = {}
        for cls, cnt in zip(unique, counts):
            key = self.class_names[int(cls)] if 0 <= int(cls) < len(self.class_names) else str(int(cls))
            dist[key] = {"count": int(cnt), "percent": round(int(cnt) * 100.0 / total, 4)}
        out = {
            "predictions": preds.tolist(),
            "distribution": dist,
            "feature_names": self.feature_names,
            "class_names": self.class_names,
            "n_rows": n
        }
        if return_proba:
            probs = np.random.rand(n, len(self.class_names))
            probs = probs / probs.sum(axis=1, keepdims=True)
            out["probabilities"] = probs.tolist()
        return out

if use_dummy:
    load_models = lambda: DummyPredictor()
    explain_instance = lambda inst, **k: {
        "feature_names": ["f1","f2","f3"],
        "values": [[[0,0,0]]],
        "expected_value": [0]
    }
    get_actionable_insight = lambda *a, **k: "No insight (dummy)."

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:5173",
        "http://127.0.0.1",
        "http://127.0.0.1:5173"
    ],
    allow_origin_regex=".*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    predictor = load_models()
    print(
        f"Successfully loaded models. Features: "
        f"{len(predictor.feature_names) if getattr(predictor,'feature_names',None) else 'unknown'}, "
        f"Classes: {len(predictor.class_names) if getattr(predictor,'class_names',None) else 'unknown'}"
    )
except Exception as e:
    print(f"--- WARNING: Could not instantiate models: {e} ---")
    traceback.print_exc()
    predictor = DummyPredictor()

FEEDBACK_FILE = ROOT / 'feedback.csv'

def _preprocess_uploaded_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    model_cols = getattr(predictor, "feature_names", None)
    if model_cols is None:
        return df.replace([np.inf,-np.inf], np.nan).fillna(0)
    for col in model_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[model_cols]
    return df.replace([np.inf,-np.inf], np.nan).fillna(0)

@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...), sample_rows: int = 100):
    try:
        contents = await file.read()
        try:
            df = pd.read_csv(BytesIO(contents), encoding='utf-8', low_memory=False)
        except Exception:
            df = pd.read_csv(BytesIO(contents), encoding='latin-1', low_memory=False)
    except Exception as e:
        raise HTTPException(400, f"Could not read uploaded CSV: {e}")

    original_row_count = len(df)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    df_processed = _preprocess_uploaded_df(df)

    try:
        result = predictor.infer(df_processed, return_proba=True)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Inference failed: {e}")

    analysis = []
    for name, data in result.get("distribution", {}).items():
        cnt = data.get("count", 0)
        pct = data.get("percent", 0)
        if isinstance(cnt, float) and (np.isnan(cnt) or np.isinf(cnt)):
            cnt = 0
        if isinstance(pct, float) and (np.isnan(pct) or np.isinf(pct)):
            pct = 0
        analysis.append({"name": name, "value": int(cnt), "percent": float(pct)})

    preview_data = (
        df.head(sample_rows)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .to_dict(orient="records")
    )
    cols = getattr(predictor, "feature_names", list(df.columns))

    response = {
        "message": "File processed successfully",
        "fileName": getattr(file, "filename", "uploaded.csv"),
        "rowCount": original_row_count,
        "analysis": analysis,
        "previewData": preview_data,
        "processedColumns": cols
    }

    return json.loads(json.dumps(response, cls=CustomEncoder))

@app.post("/api/predict_row")
async def api_predict_row(row: Dict[str, Any]):
    try:
        df = pd.DataFrame([row])
        df_processed = _preprocess_uploaded_df(df)

        res = predictor.infer(df_processed, return_proba=True)
        pred_numeric = int(res["predictions"][0])
        class_names = res["class_names"]
        prediction_label = (
            class_names[pred_numeric]
            if pred_numeric < len(class_names)
            else str(pred_numeric)
        )

        try:
            shap_payload = explain_instance(df_processed, method="auto")
        except Exception:
            shap_payload = {
                "feature_names": getattr(predictor, "feature_names", []),
                "values": [],
                "expected_value": None
            }

        explanation = []
        feature_names = shap_payload.get("feature_names", [])
        shap_values_list = shap_payload.get("values", [])

        if shap_values_list and feature_names:
            idx = pred_numeric if pred_numeric < len(shap_values_list) else 0
            class_shap_values = shap_values_list[idx]
            arr = np.asarray(class_shap_values, dtype="object")
            while arr.ndim > 1:
                arr = arr[0]
            arr = np.array(arr, dtype=float)
            row_values = df.iloc[0] if not df.empty else pd.Series({})
            temp_explanation = []
            for i, f in enumerate(feature_names):
                contrib = float(arr[i]) if i < len(arr) else 0.0
                temp_explanation.append({
                    "feature": f,
                    "value": float(row_values.get(f, 0)),
                    "contribution": contrib
                })
            temp_explanation.sort(key=lambda x: abs(x["contribution"]), reverse=True)
            explanation = temp_explanation[:6]

        final_response = {
            "prediction": pred_numeric,
            "prediction_label": prediction_label,
            "explanation": explanation,
            "probabilities": res.get("probabilities", [])[0]
                if res.get("probabilities") else None
        }

        try:
            insight = get_actionable_insight(pred_numeric, shap_payload, class_names)
            final_response["insight"] = insight
        except Exception:
            final_response["insight"] = None

        return json.loads(json.dumps(final_response, cls=CustomEncoder))

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Error in predict_row: {e}")

@app.post("/api/feedback")
async def api_feedback(payload: dict):
    try:
        rec = payload.copy()
        rec["_ts"] = datetime.datetime.utcnow().isoformat()
        df = pd.DataFrame([rec])
        if not FEEDBACK_FILE.exists():
            df.to_csv(FEEDBACK_FILE, index=False)
        else:
            df.to_csv(FEEDBACK_FILE, mode="a", header=False, index=False)
        return {"message": "Feedback received. Thank you!"}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))

@app.post("/api/chatbot")
async def api_chatbot(data: Dict[str, str]):
    msg = data.get("query", "").strip()
    if not msg:
        return {"response": "please ask a question."}

    FAQ = {
        "hey": "hello there. how can i help you with security basics?",
        "hello": "hi. feel free to ask anything about attacks or ids.",
        "hi": "hey. tell me what you want to learn.",
        "can you help me": "yes, just ask your security or ml question.",

        "what to do ddos": "block offending ip ranges, rate limit requests, and use ddos mitigation services.",
        "ddos": "a distributed denial of service attack overwhelms a target with massive fake traffic until it goes offline.",

        "what to do hulk": "throttle http request rates and enable strict rate limiting.",
        "dos hulk": "it blasts a server with nonstop random get requests to crash it.",

        "what to do goldeneye": "tighten server timeouts and drop slow connections quickly.",
        "dos goldeneye": "it sends slow incomplete http requests to drain server resources.",

        "what to do slowloris": "close idle connections quickly or use a reverse proxy.",
        "dos slowloris": "it keeps many half open connections to choke the server.",

        "what to do slowhttptest": "enforce minimum data rates and terminate slow clients.",
        "dos slowhttptest": "it stalls servers by sending extremely slow http operations.",

        "what to do heartbleed": "patch openssl immediately and regenerate any exposed keys.",
        "dos heartbleed": "a tls heartbeat flaw that leaks sensitive server memory.",

        "what to do brute ftp": "lock accounts after failures and require strong passwords.",
        "brute force ftp": "repeated password guessing to gain ftp access.",

        "what to do brute ssh": "disable password login and use ssh keys only.",
        "brute force ssh": "trying many username password combos to steal ssh access.",

        "what to do xss": "sanitize all user inputs and escape dynamic content.",
        "web attack xss": "injecting malicious scripts into websites to hijack users.",

        "what to do sql injection": "use parameterized queries and strict input validation.",
        "web attack sql injection": "inserting malicious sql queries to read or corrupt database data.",

        "what to do command injection": "never pass unsanitized user input into system commands.",
        "web attack command injection": "forcing a server to run attacker supplied system commands.",

        "what to do botnet": "isolate the device, scan for malware, block command and control traffic.",
        "botnet": "a machine gets taken over and controlled remotely as part of a larger attack network.",

        "what to do portscan": "monitor repeated probes and temporarily block scanning hosts.",
        "portscan": "probing open ports on a target to find weaknesses.",

        "what to do loit": "enable aggressive traffic filtering and use ddos protection services.",
        "ddos loit": "launching distributed traffic floods using the low orbit ion cannon tool.",

        "ids": "an intrusion detection system monitors network activity and flags malicious traffic.",
        "ips": "an intrusion prevention system detects attacks and blocks them in real time.",
        "shap": "shap explains how each feature contributes to a machine learning prediction."
    }

    low = msg.lower()

    for key, ans in sorted(FAQ.items(), key=lambda x: len(x[0]), reverse=True):
        if key in low:
            return {"response": ans}

    return {"response": "i couldn't match that with any security concept i know. try asking about attacks or what to do?"}

@app.post("/api/generate-full-report")
async def generate_full_report(file: UploadFile = File(...)):

    try:
        contents = await file.read()
        try:
            df = pd.read_csv(BytesIO(contents), encoding="utf-8", low_memory=False)
        except Exception:
            df = pd.read_csv(BytesIO(contents), encoding="latin-1", low_memory=False)

        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        df = _preprocess_uploaded_df(df)

        res = predictor.infer(df)
        class_names = res.get("class_names", [])
        preds = res.get("predictions", [])

        results = []
        for pred in preds:
            pred = int(pred)
            label = class_names[pred] if 0 <= pred < len(class_names) else str(pred)
            results.append({
                "prediction_label": label,
                "explanation": [],
                "insight": None
            })

        reports_dir = ROOT / "reports"
        reports_dir.mkdir(exist_ok=True)

        filename = f"Full_Attack_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        save_path = reports_dir / filename

        generate_attack_report(results, str(save_path))
        print("PDF GENERATED AT:", save_path)

        return FileResponse(
            path=str(save_path),
            filename="Full_Attack_Report.pdf",
            media_type="application/pdf",
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Failed to generate report: {str(e)}")
