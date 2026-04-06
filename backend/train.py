import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import shap
import joblib
import json
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "WA_Fn-UseC_-HR-Employee-Attrition.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

COLS_TO_DROP = ["EmployeeCount", "Over18", "StandardHours", "EmployeeNumber"]
CATEGORICAL_COLS = ["BusinessTravel", "Department", "EducationField", "Gender", "JobRole", "MaritalStatus", "OverTime"]


def _get_class1_shap(raw):
    """Handle both old-style list and new-style ndarray from shap."""
    if isinstance(raw, list):
        return raw[1]
    return raw[:, :, 1]


def train():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=COLS_TO_DROP)
    df["Attrition"] = (df["Attrition"] == "Yes").astype(int)

    # Compute defaults from raw (pre-encoding) values
    defaults = {}
    for col in df.columns:
        if col == "Attrition":
            continue
        if col in CATEGORICAL_COLS:
            defaults[col] = str(df[col].mode()[0])
        else:
            defaults[col] = int(df[col].median())

    # Label-encode categoricals
    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df.drop(columns=["Attrition"])
    y = df["Attrition"]
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    explainer = shap.TreeExplainer(model)
    raw_shap = explainer.shap_values(X_train)

    # Handle shap 0.46+ which may return an Explanation object
    if hasattr(raw_shap, 'values'):
        raw_shap = raw_shap.values

    class1_shap = _get_class1_shap(raw_shap)

    global_shap = {
        name: round(float(np.mean(np.abs(class1_shap[:, i]))), 6)
        for i, name in enumerate(feature_names)
    }
    global_shap_sorted = dict(sorted(global_shap.items(), key=lambda x: x[1], reverse=True))

    joblib.dump(model, os.path.join(MODEL_DIR, "rf_model.pkl"))
    joblib.dump(explainer, os.path.join(MODEL_DIR, "shap_explainer.pkl"))
    joblib.dump(encoders, os.path.join(MODEL_DIR, "encoders.pkl"))
    joblib.dump(feature_names, os.path.join(MODEL_DIR, "feature_names.pkl"))

    with open(os.path.join(MODEL_DIR, "feature_defaults.json"), "w") as f:
        json.dump(defaults, f)
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f)
    with open(os.path.join(MODEL_DIR, "global_shap.json"), "w") as f:
        json.dump(global_shap_sorted, f)

    print(f"Training complete. Accuracy: {metrics['accuracy']}, F1: {metrics['f1']}")
    return metrics


if __name__ == "__main__":
    train()
