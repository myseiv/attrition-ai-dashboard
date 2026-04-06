import json
import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
CATEGORICAL_COLS = ["BusinessTravel", "Department", "EducationField", "Gender", "JobRole", "MaritalStatus", "OverTime"]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Module-level state populated at startup
model = None
explainer = None
encoders = None
feature_names = None
feature_defaults = None
global_shap_data = None
metrics_data = None
claude_summary_cache: Optional[str] = None

FALLBACK_SUMMARY = (
    "The top predictors of employee attrition in this dataset are overtime work, "
    "monthly income, and job satisfaction. Employees who regularly work overtime "
    "and report lower job satisfaction are significantly more likely to leave."
)


def _get_class1_shap(raw):
    if isinstance(raw, list):
        return raw[1]
    return raw[:, :, 1]


def load_artifacts():
    global model, explainer, encoders, feature_names, feature_defaults, global_shap_data, metrics_data
    if not os.path.exists(os.path.join(MODEL_DIR, "rf_model.pkl")):
        print("Model not found — running training pipeline...")
        from train import train
        train()
    model = joblib.load(os.path.join(MODEL_DIR, "rf_model.pkl"))
    explainer = joblib.load(os.path.join(MODEL_DIR, "shap_explainer.pkl"))
    encoders = joblib.load(os.path.join(MODEL_DIR, "encoders.pkl"))
    feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
    with open(os.path.join(MODEL_DIR, "feature_defaults.json")) as f:
        feature_defaults = json.load(f)
    with open(os.path.join(MODEL_DIR, "global_shap.json")) as f:
        global_shap_data = json.load(f)
    with open(os.path.join(MODEL_DIR, "metrics.json")) as f:
        metrics_data = json.load(f)


@app.on_event("startup")
async def startup_event():
    load_artifacts()


# Also load at import time so TestClient (which may skip startup) works in tests
load_artifacts()


class PredictRequest(BaseModel):
    OverTime: str
    Age: int
    MonthlyIncome: int
    JobSatisfaction: int
    YearsAtCompany: int
    WorkLifeBalance: int
    JobLevel: int
    DistanceFromHome: int
    NumCompaniesWorked: int
    StockOptionLevel: int


@app.post("/predict")
def predict(req: PredictRequest):
    # Start from defaults, overlay user inputs
    row = dict(feature_defaults)
    for key, val in req.dict().items():
        row[key] = val

    # Encode categoricals
    for col in CATEGORICAL_COLS:
        if col in row and col in encoders:
            try:
                row[col] = int(encoders[col].transform([str(row[col])])[0])
            except ValueError:
                raise HTTPException(status_code=422, detail=f"Invalid value for {col}: {row[col]}")

    # Build DataFrame in training feature order
    df_row = pd.DataFrame([[row[col] for col in feature_names]], columns=feature_names)

    # Predict
    proba = model.predict_proba(df_row)[0]
    leave_prob = float(proba[1])
    prediction = "Leave" if leave_prob >= 0.5 else "Stay"
    confidence = round(leave_prob if prediction == "Leave" else 1 - leave_prob, 4)

    # Local SHAP
    raw_shap = explainer.shap_values(df_row)
    if hasattr(raw_shap, 'values'):
        raw_shap = raw_shap.values
    local_shap = _get_class1_shap(raw_shap)[0]

    user_vals = req.dict()
    shap_list = [
        {
            "feature": feature_names[i],
            "value": str(user_vals.get(feature_names[i], feature_defaults.get(feature_names[i]))),
            "shap": round(float(local_shap[i]), 4),
            "direction": "leave" if local_shap[i] > 0 else "stay",
        }
        for i in range(len(feature_names))
    ]
    shap_list.sort(key=lambda x: abs(x["shap"]), reverse=True)

    return {"prediction": prediction, "confidence": confidence, "shap_values": shap_list[:10]}


@app.get("/global-importance")
def global_importance():
    global claude_summary_cache

    features = [{"feature": k, "importance": v} for k, v in global_shap_data.items()]

    if claude_summary_cache is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            try:
                import anthropic
                top5_str = "\n".join(
                    f"- {f['feature']}: {f['importance']:.4f}" for f in features[:5]
                )
                client_ai = anthropic.Anthropic(api_key=api_key)
                message = client_ai.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=200,
                    system=(
                        "You are a data analyst explaining a machine learning model to a business stakeholder. "
                        "Given the following feature importances from an employee attrition model, write 2-3 sentences "
                        "in plain English explaining what the model has learned. Be specific. Do not use jargon. "
                        "Do not mention SHAP or Random Forest. Focus on what the business should pay attention to."
                    ),
                    messages=[{"role": "user", "content": f"Top 5 features by importance:\n{top5_str}"}],
                )
                claude_summary_cache = message.content[0].text
            except Exception:
                claude_summary_cache = FALLBACK_SUMMARY
        else:
            claude_summary_cache = FALLBACK_SUMMARY

    return {"features": features, "summary": claude_summary_cache}


@app.get("/model-metrics")
def model_metrics():
    return metrics_data


@app.post("/train")
def retrain():
    from train import train
    new_metrics = train()
    load_artifacts()
    return new_metrics
