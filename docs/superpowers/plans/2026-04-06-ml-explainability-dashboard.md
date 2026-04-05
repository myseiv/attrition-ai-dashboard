# ML Explainability Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a backend-first ML explainability dashboard — FastAPI + scikit-learn + SHAP backend, Next.js frontend — that trains a Random Forest on the IBM HR Attrition dataset and explains predictions with SHAP values.

**Architecture:** Backend-first. Train the ML pipeline and verify all API endpoints before touching the frontend. The FastAPI backend serves four endpoints; the Next.js frontend proxies all `/api/*` requests to it via `next.config.js` rewrites.

**Tech Stack:** Python 3.11, FastAPI, scikit-learn, SHAP, joblib, pandas, pytest — Next.js 14 (app router), TypeScript, Tailwind CSS, Recharts

---

## File Map

```
proj5/
├── .gitignore
├── .env.example
├── backend/
│   ├── main.py              # FastAPI app — loads model at startup, serves 4 endpoints
│   ├── train.py             # Training pipeline — run once, writes model/ artifacts
│   ├── test_api.py          # pytest tests for all 4 endpoints
│   ├── requirements.txt
│   └── data/
│       └── WA_Fn-UseC_-HR-Employee-Attrition.csv   # bundled dataset
└── frontend/
    ├── next.config.js       # rewrites /api/* → http://localhost:8000/*
    ├── app/
    │   ├── layout.tsx       # root layout with NavBar
    │   ├── page.tsx         # Predict + Explain (two-column)
    │   ├── whatif/
    │   │   └── page.tsx     # What-If Explorer (live sliders)
    │   └── global/
    │       └── page.tsx     # Global Importance + Model Metrics
    ├── components/
    │   ├── NavBar.tsx
    │   ├── PredictionResult.tsx
    │   ├── ShapChart.tsx
    │   └── ConfusionMatrix.tsx
    └── lib/
        ├── api.ts           # typed fetch wrappers
        └── types.ts         # shared TypeScript interfaces
```

---

## Task 1: Project Scaffolding

**Files:**
- Create: `.gitignore`
- Create: `.env.example`
- Create: `backend/data/` directory

- [ ] **Step 1: Create .gitignore**

```
# Python
backend/model/
backend/__pycache__/
backend/.pytest_cache/
*.pyc
*.pyo
*.egg-info/
.venv/
venv/

# Node
frontend/node_modules/
frontend/.next/
frontend/.env.local

# Env
.env

# Brainstorm artifacts
.superpowers/
```

Save as `proj5/.gitignore`.

- [ ] **Step 2: Create .env.example**

```
ANTHROPIC_API_KEY=your_key_here
```

Save as `proj5/.env.example`.

- [ ] **Step 3: Create backend/data directory**

```bash
mkdir -p backend/data
```

- [ ] **Step 4: Download the dataset**

Download the IBM HR Employee Attrition dataset from Kaggle:
https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

Place the file at `backend/data/WA_Fn-UseC_-HR-Employee-Attrition.csv`.

Verify it exists and has the right shape:
```bash
python -c "import pandas as pd; df = pd.read_csv('backend/data/WA_Fn-UseC_-HR-Employee-Attrition.csv'); print(df.shape)"
```
Expected output: `(1470, 35)`

- [ ] **Step 5: Commit**

```bash
git add .gitignore .env.example backend/data/WA_Fn-UseC_-HR-Employee-Attrition.csv
git commit -m "chore: scaffold project structure and bundle dataset"
```

---

## Task 2: Backend Dependencies

**Files:**
- Create: `backend/requirements.txt`

- [ ] **Step 1: Create requirements.txt**

```
fastapi==0.104.1
uvicorn==0.24.0
scikit-learn==1.3.2
shap==0.43.0
pandas==2.1.3
numpy==1.26.2
joblib==1.3.2
anthropic==0.28.0
httpx==0.25.2
pytest==7.4.3
python-multipart==0.0.6
```

Save as `backend/requirements.txt`.

- [ ] **Step 2: Create and activate a virtual environment**

```bash
cd backend
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

- [ ] **Step 3: Install dependencies**

```bash
pip install -r requirements.txt
```

Expected: no errors. `shap`, `sklearn`, `fastapi` all installed.

- [ ] **Step 4: Commit**

```bash
git add backend/requirements.txt
git commit -m "chore: add backend requirements"
```

---

## Task 3: Training Pipeline

**Files:**
- Create: `backend/train.py`

- [ ] **Step 1: Create train.py**

```python
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
```

Save as `backend/train.py`.

- [ ] **Step 2: Run the training pipeline**

```bash
cd backend
python train.py
```

Expected output (values will vary slightly):
```
Training complete. Accuracy: 0.8639, F1: 0.5...
```

- [ ] **Step 3: Verify model artifacts were created**

```bash
ls model/
```

Expected files: `rf_model.pkl`, `shap_explainer.pkl`, `encoders.pkl`, `feature_names.pkl`, `feature_defaults.json`, `metrics.json`, `global_shap.json`

- [ ] **Step 4: Verify global_shap.json looks sensible**

```bash
python -c "import json; d = json.load(open('model/global_shap.json')); print(list(d.items())[:5])"
```

Expected: OverTime or MonthlyIncome near the top with non-zero values.

- [ ] **Step 5: Commit**

```bash
cd ..
git add backend/train.py
git commit -m "feat: add training pipeline with SHAP and model persistence"
```

---

## Task 4: FastAPI App Foundation

**Files:**
- Create: `backend/main.py`
- Create: `backend/test_api.py` (skeleton)

- [ ] **Step 1: Create main.py**

```python
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
```

Save as `backend/main.py`.

- [ ] **Step 2: Verify the app starts without errors**

```bash
cd backend
uvicorn main:app --reload --port 8000
```

Expected: `Application startup complete.` with no errors. Stop with Ctrl+C.

- [ ] **Step 3: Create test_api.py skeleton**

```python
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

VALID_PAYLOAD = {
    "OverTime": "Yes",
    "Age": 35,
    "MonthlyIncome": 5000,
    "JobSatisfaction": 2,
    "YearsAtCompany": 3,
    "WorkLifeBalance": 2,
    "JobLevel": 2,
    "DistanceFromHome": 10,
    "NumCompaniesWorked": 4,
    "StockOptionLevel": 0,
}
```

Save as `backend/test_api.py`.

- [ ] **Step 4: Commit**

```bash
cd ..
git add backend/main.py backend/test_api.py
git commit -m "feat: add FastAPI app skeleton with startup model loading"
```

---

## Task 5: POST /predict Endpoint

**Files:**
- Modify: `backend/main.py`
- Modify: `backend/test_api.py`

- [ ] **Step 1: Add failing test to test_api.py**

Append to `backend/test_api.py`:

```python
def test_predict_valid():
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in ("Leave", "Stay")
    assert 0.0 <= data["confidence"] <= 1.0
    assert len(data["shap_values"]) == 10
    first = data["shap_values"][0]
    assert all(k in first for k in ("feature", "value", "shap", "direction"))
    assert first["direction"] in ("leave", "stay")


def test_predict_missing_field():
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "Age"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd backend
pytest test_api.py::test_predict_valid test_api.py::test_predict_missing_field -v
```

Expected: both FAIL — `test_predict_valid` with 404 or 405, `test_predict_missing_field` also 404/405.

- [ ] **Step 3: Add PredictRequest model and /predict endpoint to main.py**

Add after `FALLBACK_SUMMARY` definition in `backend/main.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest test_api.py::test_predict_valid test_api.py::test_predict_missing_field -v
```

Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
cd ..
git add backend/main.py backend/test_api.py
git commit -m "feat: add POST /predict endpoint with SHAP explanations"
```

---

## Task 6: GET /global-importance Endpoint

**Files:**
- Modify: `backend/main.py`
- Modify: `backend/test_api.py`

- [ ] **Step 1: Add failing test**

Append to `backend/test_api.py`:

```python
def test_global_importance():
    response = client.get("/global-importance")
    assert response.status_code == 200
    data = response.json()
    assert "features" in data
    assert "summary" in data
    assert len(data["features"]) > 0
    first = data["features"][0]
    assert "feature" in first and "importance" in first
    assert isinstance(data["summary"], str) and len(data["summary"]) > 10
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd backend
pytest test_api.py::test_global_importance -v
```

Expected: FAIL with 404.

- [ ] **Step 3: Add /global-importance endpoint to main.py**

Append to `backend/main.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest test_api.py::test_global_importance -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd ..
git add backend/main.py backend/test_api.py
git commit -m "feat: add GET /global-importance endpoint with Claude fallback"
```

---

## Task 7: GET /model-metrics Endpoint + Full Test Run

**Files:**
- Modify: `backend/main.py`
- Modify: `backend/test_api.py`

- [ ] **Step 1: Add failing test**

Append to `backend/test_api.py`:

```python
def test_model_metrics():
    response = client.get("/model-metrics")
    assert response.status_code == 200
    data = response.json()
    assert "accuracy" in data
    assert "precision" in data
    assert "recall" in data
    assert "f1" in data
    assert "confusion_matrix" in data
    assert len(data["confusion_matrix"]) == 2
    assert len(data["confusion_matrix"][0]) == 2
    assert 0.0 <= data["accuracy"] <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd backend
pytest test_api.py::test_model_metrics -v
```

Expected: FAIL with 404.

- [ ] **Step 3: Add /model-metrics and /train endpoints to main.py**

Append to `backend/main.py`:

```python
@app.get("/model-metrics")
def model_metrics():
    return metrics_data


@app.post("/train")
def retrain():
    from train import train
    new_metrics = train()
    load_artifacts()
    return new_metrics
```

- [ ] **Step 4: Run the full test suite**

```bash
pytest test_api.py -v
```

Expected: all 4 tests PASS.
```
test_api.py::test_predict_valid PASSED
test_api.py::test_predict_missing_field PASSED
test_api.py::test_global_importance PASSED
test_api.py::test_model_metrics PASSED
```

- [ ] **Step 5: Commit**

```bash
cd ..
git add backend/main.py backend/test_api.py
git commit -m "feat: add GET /model-metrics and POST /train endpoints — all tests pass"
```

---

## Task 8: Frontend Scaffolding

**Files:**
- Create: `frontend/` (scaffolded by create-next-app)
- Modify: `frontend/next.config.js`

- [ ] **Step 1: Scaffold Next.js app**

```bash
cd proj5
npx create-next-app@14 frontend --typescript --tailwind --eslint --app --no-src-dir --import-alias "@/*" --no-git
```

When prompted: accept all defaults.

- [ ] **Step 2: Install Recharts**

```bash
cd frontend
npm install recharts
npm install --save-dev @types/recharts
```

- [ ] **Step 3: Replace next.config.js with API proxy**

```javascript
/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://localhost:8000/:path*',
      },
    ]
  },
}

module.exports = nextConfig
```

Save as `frontend/next.config.js`.

- [ ] **Step 4: Delete boilerplate**

Delete `frontend/app/page.tsx` content (we'll replace it in Task 13).
Delete `frontend/app/globals.css` content and replace with:

```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

- [ ] **Step 5: Verify frontend starts**

```bash
cd frontend
npm run dev
```

Expected: `ready - started server on 0.0.0.0:3000`. Open http://localhost:3000 — blank page with no errors. Stop with Ctrl+C.

- [ ] **Step 6: Commit**

```bash
cd ..
git add frontend/
git commit -m "chore: scaffold Next.js frontend with Tailwind and Recharts"
```

---

## Task 9: Shared Types, API Client, and NavBar

**Files:**
- Create: `frontend/lib/types.ts`
- Create: `frontend/lib/api.ts`
- Create: `frontend/components/NavBar.tsx`
- Modify: `frontend/app/layout.tsx`

- [ ] **Step 1: Create frontend/lib/types.ts**

```typescript
export interface ShapValue {
  feature: string
  value: string
  shap: number
  direction: 'leave' | 'stay'
}

export interface PredictResponse {
  prediction: 'Leave' | 'Stay'
  confidence: number
  shap_values: ShapValue[]
}

export interface GlobalFeature {
  feature: string
  importance: number
}

export interface GlobalImportanceResponse {
  features: GlobalFeature[]
  summary: string
}

export interface ModelMetricsResponse {
  accuracy: number
  precision: number
  recall: number
  f1: number
  confusion_matrix: [[number, number], [number, number]]
}
```

- [ ] **Step 2: Create frontend/lib/api.ts**

```typescript
import { PredictResponse, GlobalImportanceResponse, ModelMetricsResponse } from './types'

const BASE = '/api'

export async function predict(data: Record<string, string | number>): Promise<PredictResponse> {
  const res = await fetch(`${BASE}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function getGlobalImportance(): Promise<GlobalImportanceResponse> {
  const res = await fetch(`${BASE}/global-importance`)
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function getModelMetrics(): Promise<ModelMetricsResponse> {
  const res = await fetch(`${BASE}/model-metrics`)
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}
```

- [ ] **Step 3: Create frontend/components/NavBar.tsx**

```typescript
'use client'
import Link from 'next/link'
import { usePathname } from 'next/navigation'

const links = [
  { href: '/', label: 'Predict' },
  { href: '/whatif', label: 'What-If' },
  { href: '/global', label: 'Global Importance' },
]

export default function NavBar() {
  const pathname = usePathname()
  return (
    <nav className="bg-slate-900 text-white px-6 py-3 flex items-center gap-8">
      <span className="font-bold text-white tracking-tight">⚡ AttritionAI</span>
      <div className="flex gap-6">
        {links.map(({ href, label }) => (
          <Link
            key={href}
            href={href}
            className={`text-sm transition-colors ${
              pathname === href
                ? 'text-white border-b-2 border-blue-400 pb-0.5'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            {label}
          </Link>
        ))}
      </div>
    </nav>
  )
}
```

- [ ] **Step 4: Update frontend/app/layout.tsx**

```typescript
import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import NavBar from '@/components/NavBar'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'AttritionAI — ML Explainability Dashboard',
  description: 'Predict and explain employee attrition with SHAP',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className={`${inter.className} bg-gray-50 min-h-screen`}>
        <NavBar />
        <main className="max-w-7xl mx-auto px-6 py-8">{children}</main>
      </body>
    </html>
  )
}
```

- [ ] **Step 5: Verify NavBar renders**

Start backend: `cd backend && uvicorn main:app --port 8000`
Start frontend in a second terminal: `cd frontend && npm run dev`
Open http://localhost:3000 — nav bar visible with Predict / What-If / Global Importance links.

- [ ] **Step 6: Commit**

```bash
git add frontend/lib/ frontend/components/NavBar.tsx frontend/app/layout.tsx
git commit -m "feat: add shared types, API client, and NavBar"
```

---

## Task 10: PredictionResult Component

**Files:**
- Create: `frontend/components/PredictionResult.tsx`

- [ ] **Step 1: Create PredictionResult.tsx**

```typescript
import { PredictResponse } from '@/lib/types'

interface Props {
  result: PredictResponse
}

export default function PredictionResult({ result }: Props) {
  const isLeave = result.prediction === 'Leave'
  const pct = Math.round(result.confidence * 100)

  return (
    <div className="flex items-center gap-4 p-4 rounded-lg border bg-white">
      <div
        className={`px-5 py-2 rounded-md text-lg font-bold tracking-wide ${
          isLeave
            ? 'bg-amber-100 text-amber-800 border border-amber-300'
            : 'bg-green-100 text-green-800 border border-green-300'
        }`}
      >
        {isLeave ? '⚠ LEAVE' : '✓ STAY'}
      </div>
      <div>
        <p className="text-2xl font-semibold text-slate-800">{pct}% confidence</p>
        <p className="text-sm text-slate-500">
          {isLeave
            ? 'This employee is likely to leave'
            : 'This employee is likely to stay'}
        </p>
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/components/PredictionResult.tsx
git commit -m "feat: add PredictionResult component"
```

---

## Task 11: ShapChart Component

**Files:**
- Create: `frontend/components/ShapChart.tsx`

- [ ] **Step 1: Create ShapChart.tsx**

```typescript
'use client'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ReferenceLine, ResponsiveContainer, Cell } from 'recharts'
import { ShapValue } from '@/lib/types'

interface Props {
  shapValues: ShapValue[]
}

const CustomTooltip = ({ active, payload }: any) => {
  if (!active || !payload?.length) return null
  const d = payload[0].payload
  return (
    <div className="bg-white border border-slate-200 rounded p-2 text-sm shadow">
      <p className="font-medium">{d.feature}</p>
      <p className="text-slate-500">Value: {d.value}</p>
      <p className={d.shap > 0 ? 'text-red-600' : 'text-blue-600'}>
        SHAP: {d.shap > 0 ? '+' : ''}{d.shap.toFixed(4)}
      </p>
    </div>
  )
}

export default function ShapChart({ shapValues }: Props) {
  // Sort by shap value for visual clarity (most negative at top, most positive at bottom)
  const data = [...shapValues].sort((a, b) => a.shap - b.shap)

  return (
    <div className="bg-white rounded-lg border p-4">
      <h3 className="text-sm font-semibold text-slate-700 mb-1">Why this prediction?</h3>
      <p className="text-xs text-slate-400 mb-3">
        <span className="text-blue-500 font-medium">Blue</span> = pushes toward Stay &nbsp;|&nbsp;
        <span className="text-red-500 font-medium">Red</span> = pushes toward Leave
      </p>
      <ResponsiveContainer width="100%" height={280}>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 0, right: 20, left: 130, bottom: 0 }}
        >
          <XAxis type="number" domain={['auto', 'auto']} tick={{ fontSize: 11 }} />
          <YAxis
            type="category"
            dataKey="feature"
            tick={{ fontSize: 12 }}
            width={120}
          />
          <Tooltip content={<CustomTooltip />} />
          <ReferenceLine x={0} stroke="#94a3b8" />
          <Bar dataKey="shap" radius={2}>
            {data.map((entry, index) => (
              <Cell
                key={index}
                fill={entry.shap > 0 ? '#ef4444' : '#3b82f6'}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/components/ShapChart.tsx
git commit -m "feat: add ShapChart component with Recharts horizontal bar chart"
```

---

## Task 12: ConfusionMatrix Component

**Files:**
- Create: `frontend/components/ConfusionMatrix.tsx`

- [ ] **Step 1: Create ConfusionMatrix.tsx**

```typescript
interface Props {
  matrix: [[number, number], [number, number]]
}

// matrix layout: [[TN, FP], [FN, TP]]
const CELLS = [
  { label: 'True Negative', row: 0, col: 0, color: 'bg-green-100 text-green-800 border-green-300' },
  { label: 'False Positive', row: 0, col: 1, color: 'bg-red-100 text-red-800 border-red-300' },
  { label: 'False Negative', row: 1, col: 0, color: 'bg-red-100 text-red-800 border-red-300' },
  { label: 'True Positive', row: 1, col: 1, color: 'bg-green-100 text-green-800 border-green-300' },
]

export default function ConfusionMatrix({ matrix }: Props) {
  return (
    <div>
      <div className="flex gap-1 mb-1">
        <div className="w-24" />
        <div className="flex-1 text-center text-xs text-slate-500 font-medium">Predicted: Stay</div>
        <div className="flex-1 text-center text-xs text-slate-500 font-medium">Predicted: Leave</div>
      </div>
      {[0, 1].map((row) => (
        <div key={row} className="flex gap-1 mb-1">
          <div className="w-24 flex items-center justify-end pr-2 text-xs text-slate-500 font-medium">
            {row === 0 ? 'Actual: Stay' : 'Actual: Leave'}
          </div>
          {[0, 1].map((col) => {
            const cell = CELLS[row * 2 + col]
            return (
              <div
                key={col}
                className={`flex-1 border rounded-lg p-4 text-center ${cell.color}`}
              >
                <p className="text-2xl font-bold">{matrix[row][col]}</p>
                <p className="text-xs mt-1 font-medium">{cell.label}</p>
              </div>
            )
          })}
        </div>
      ))}
    </div>
  )
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/components/ConfusionMatrix.tsx
git commit -m "feat: add ConfusionMatrix heatmap component"
```

---

## Task 13: Predict Page

**Files:**
- Modify: `frontend/app/page.tsx`

- [ ] **Step 1: Create frontend/app/page.tsx**

```typescript
'use client'
import { useState } from 'react'
import { predict } from '@/lib/api'
import { PredictResponse } from '@/lib/types'
import PredictionResult from '@/components/PredictionResult'
import ShapChart from '@/components/ShapChart'

const DEFAULTS = {
  OverTime: 'No',
  Age: 36,
  MonthlyIncome: 6500,
  JobSatisfaction: 3,
  YearsAtCompany: 5,
  WorkLifeBalance: 3,
  JobLevel: 2,
  DistanceFromHome: 9,
  NumCompaniesWorked: 2,
  StockOptionLevel: 1,
}

export default function PredictPage() {
  const [form, setForm] = useState<Record<string, string | number>>(DEFAULTS)
  const [result, setResult] = useState<PredictResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    try {
      const res = await predict(form)
      setResult(res)
    } catch (err: any) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const set = (key: string, val: string | number) => setForm((f) => ({ ...f, [key]: val }))

  return (
    <div>
      <h1 className="text-2xl font-bold text-slate-800 mb-1">Predict Attrition</h1>
      <p className="text-slate-500 text-sm mb-6">Fill in employee attributes to get a prediction with explanation.</p>

      <div className="grid grid-cols-2 gap-8">
        {/* Left: Form */}
        <form onSubmit={handleSubmit} className="bg-white rounded-lg border p-6 space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <label className="block">
              <span className="text-xs font-medium text-slate-600 uppercase tracking-wide">Overtime</span>
              <select className="mt-1 w-full border rounded-md px-3 py-2 text-sm" value={form.OverTime as string} onChange={e => set('OverTime', e.target.value)}>
                <option>Yes</option><option>No</option>
              </select>
            </label>
            <label className="block">
              <span className="text-xs font-medium text-slate-600 uppercase tracking-wide">Age</span>
              <input type="number" min={18} max={60} className="mt-1 w-full border rounded-md px-3 py-2 text-sm" value={form.Age as number} onChange={e => set('Age', Number(e.target.value))} />
            </label>
            <label className="block">
              <span className="text-xs font-medium text-slate-600 uppercase tracking-wide">Monthly Income ($)</span>
              <input type="number" min={1000} max={20000} step={100} className="mt-1 w-full border rounded-md px-3 py-2 text-sm" value={form.MonthlyIncome as number} onChange={e => set('MonthlyIncome', Number(e.target.value))} />
            </label>
            <label className="block">
              <span className="text-xs font-medium text-slate-600 uppercase tracking-wide">Job Satisfaction (1–4)</span>
              <select className="mt-1 w-full border rounded-md px-3 py-2 text-sm" value={form.JobSatisfaction as number} onChange={e => set('JobSatisfaction', Number(e.target.value))}>
                <option value={1}>1 – Low</option><option value={2}>2 – Medium</option><option value={3}>3 – High</option><option value={4}>4 – Very High</option>
              </select>
            </label>
            <label className="block">
              <span className="text-xs font-medium text-slate-600 uppercase tracking-wide">Years at Company</span>
              <input type="number" min={0} max={40} className="mt-1 w-full border rounded-md px-3 py-2 text-sm" value={form.YearsAtCompany as number} onChange={e => set('YearsAtCompany', Number(e.target.value))} />
            </label>
            <label className="block">
              <span className="text-xs font-medium text-slate-600 uppercase tracking-wide">Work-Life Balance (1–4)</span>
              <select className="mt-1 w-full border rounded-md px-3 py-2 text-sm" value={form.WorkLifeBalance as number} onChange={e => set('WorkLifeBalance', Number(e.target.value))}>
                <option value={1}>1 – Bad</option><option value={2}>2 – Good</option><option value={3}>3 – Better</option><option value={4}>4 – Best</option>
              </select>
            </label>
            <label className="block">
              <span className="text-xs font-medium text-slate-600 uppercase tracking-wide">Job Level (1–5)</span>
              <select className="mt-1 w-full border rounded-md px-3 py-2 text-sm" value={form.JobLevel as number} onChange={e => set('JobLevel', Number(e.target.value))}>
                {[1,2,3,4,5].map(v => <option key={v} value={v}>{v}</option>)}
              </select>
            </label>
            <label className="block">
              <span className="text-xs font-medium text-slate-600 uppercase tracking-wide">Distance from Home (km)</span>
              <input type="number" min={1} max={30} className="mt-1 w-full border rounded-md px-3 py-2 text-sm" value={form.DistanceFromHome as number} onChange={e => set('DistanceFromHome', Number(e.target.value))} />
            </label>
            <label className="block">
              <span className="text-xs font-medium text-slate-600 uppercase tracking-wide">Companies Worked At</span>
              <input type="number" min={0} max={9} className="mt-1 w-full border rounded-md px-3 py-2 text-sm" value={form.NumCompaniesWorked as number} onChange={e => set('NumCompaniesWorked', Number(e.target.value))} />
            </label>
            <label className="block">
              <span className="text-xs font-medium text-slate-600 uppercase tracking-wide">Stock Option Level (0–3)</span>
              <select className="mt-1 w-full border rounded-md px-3 py-2 text-sm" value={form.StockOptionLevel as number} onChange={e => set('StockOptionLevel', Number(e.target.value))}>
                {[0,1,2,3].map(v => <option key={v} value={v}>{v}</option>)}
              </select>
            </label>
          </div>
          <button type="submit" disabled={loading} className="w-full bg-slate-800 hover:bg-slate-700 disabled:opacity-50 text-white rounded-md py-2.5 text-sm font-medium transition-colors">
            {loading ? 'Predicting...' : 'Predict'}
          </button>
        </form>

        {/* Right: Results */}
        <div className="space-y-4">
          {error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-700 text-sm">{error}</div>
          )}
          {!result && !error && (
            <div className="bg-white rounded-lg border p-8 text-center text-slate-400">
              <p className="text-4xl mb-2">🔍</p>
              <p className="text-sm">Fill in the form and click Predict to see the explanation.</p>
            </div>
          )}
          {result && (
            <>
              <PredictionResult result={result} />
              <ShapChart shapValues={result.shap_values} />
            </>
          )}
        </div>
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Test the predict page manually**

With backend running on port 8000 and frontend on port 3000:
- Open http://localhost:3000
- Click Predict with default values
- Verify: result badge appears (Stay/Leave), SHAP chart renders with coloured bars, no console errors

- [ ] **Step 3: Commit**

```bash
git add frontend/app/page.tsx
git commit -m "feat: add Predict page with form and SHAP chart"
```

---

## Task 14: What-If Explorer Page

**Files:**
- Create: `frontend/app/whatif/page.tsx`

- [ ] **Step 1: Create frontend/app/whatif/page.tsx**

```typescript
'use client'
import { useState, useEffect, useRef } from 'react'
import { predict } from '@/lib/api'
import { PredictResponse } from '@/lib/types'
import PredictionResult from '@/components/PredictionResult'
import ShapChart from '@/components/ShapChart'

const DEFAULTS = {
  OverTime: 'No',
  Age: 36,
  MonthlyIncome: 6500,
  JobSatisfaction: 3,
  YearsAtCompany: 5,
  WorkLifeBalance: 3,
  JobLevel: 2,
  DistanceFromHome: 9,
  NumCompaniesWorked: 2,
  StockOptionLevel: 1,
}

export default function WhatIfPage() {
  const [form, setForm] = useState<Record<string, string | number>>(DEFAULTS)
  const [result, setResult] = useState<PredictResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const debounceRef = useRef<NodeJS.Timeout | null>(null)

  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(async () => {
      setLoading(true)
      try {
        const res = await predict(form)
        setResult(res)
      } catch {
        // silently ignore during live updates
      } finally {
        setLoading(false)
      }
    }, 300)
    return () => { if (debounceRef.current) clearTimeout(debounceRef.current) }
  }, [form])

  const set = (key: string, val: string | number) => setForm((f) => ({ ...f, [key]: val }))

  const Slider = ({ label, name, min, max, step = 1 }: { label: string; name: string; min: number; max: number; step?: number }) => (
    <label className="block">
      <div className="flex justify-between items-center mb-1">
        <span className="text-xs font-medium text-slate-600 uppercase tracking-wide">{label}</span>
        <span className="text-sm font-semibold text-slate-800">{form[name]}</span>
      </div>
      <input
        type="range" min={min} max={max} step={step}
        value={form[name] as number}
        onChange={e => set(name, Number(e.target.value))}
        className="w-full accent-slate-700"
      />
    </label>
  )

  return (
    <div>
      <h1 className="text-2xl font-bold text-slate-800 mb-1">What-If Explorer</h1>
      <p className="text-slate-500 text-sm mb-6">Adjust sliders to see how the prediction changes in real time.</p>

      <div className="grid grid-cols-2 gap-8">
        {/* Left: Controls */}
        <div className="bg-white rounded-lg border p-6 space-y-5">
          <label className="block">
            <span className="text-xs font-medium text-slate-600 uppercase tracking-wide">Overtime</span>
            <select className="mt-1 w-full border rounded-md px-3 py-2 text-sm" value={form.OverTime as string} onChange={e => set('OverTime', e.target.value)}>
              <option>Yes</option><option>No</option>
            </select>
          </label>
          <Slider label="Age" name="Age" min={18} max={60} />
          <Slider label="Monthly Income ($)" name="MonthlyIncome" min={1000} max={20000} step={100} />
          <label className="block">
            <span className="text-xs font-medium text-slate-600 uppercase tracking-wide">Job Satisfaction</span>
            <select className="mt-1 w-full border rounded-md px-3 py-2 text-sm" value={form.JobSatisfaction as number} onChange={e => set('JobSatisfaction', Number(e.target.value))}>
              <option value={1}>1 – Low</option><option value={2}>2 – Medium</option><option value={3}>3 – High</option><option value={4}>4 – Very High</option>
            </select>
          </label>
          <Slider label="Years at Company" name="YearsAtCompany" min={0} max={40} />
          <label className="block">
            <span className="text-xs font-medium text-slate-600 uppercase tracking-wide">Work-Life Balance</span>
            <select className="mt-1 w-full border rounded-md px-3 py-2 text-sm" value={form.WorkLifeBalance as number} onChange={e => set('WorkLifeBalance', Number(e.target.value))}>
              <option value={1}>1 – Bad</option><option value={2}>2 – Good</option><option value={3}>3 – Better</option><option value={4}>4 – Best</option>
            </select>
          </label>
          <label className="block">
            <span className="text-xs font-medium text-slate-600 uppercase tracking-wide">Job Level</span>
            <select className="mt-1 w-full border rounded-md px-3 py-2 text-sm" value={form.JobLevel as number} onChange={e => set('JobLevel', Number(e.target.value))}>
              {[1,2,3,4,5].map(v => <option key={v} value={v}>{v}</option>)}
            </select>
          </label>
          <Slider label="Distance from Home (km)" name="DistanceFromHome" min={1} max={30} />
          <Slider label="Companies Worked At" name="NumCompaniesWorked" min={0} max={9} />
          <label className="block">
            <span className="text-xs font-medium text-slate-600 uppercase tracking-wide">Stock Option Level</span>
            <select className="mt-1 w-full border rounded-md px-3 py-2 text-sm" value={form.StockOptionLevel as number} onChange={e => set('StockOptionLevel', Number(e.target.value))}>
              {[0,1,2,3].map(v => <option key={v} value={v}>{v}</option>)}
            </select>
          </label>
        </div>

        {/* Right: Live result */}
        <div className="space-y-4">
          {loading && !result && (
            <div className="bg-white rounded-lg border p-8 text-center text-slate-400 text-sm">Loading...</div>
          )}
          {result && (
            <>
              <div className="relative">
                {loading && <div className="absolute inset-0 bg-white/60 rounded-lg z-10" />}
                <PredictionResult result={result} />
              </div>
              <ShapChart shapValues={result.shap_values} />
            </>
          )}
        </div>
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Test the what-if page manually**

Open http://localhost:3000/whatif.
- Verify: prediction loads immediately on page open
- Move the Age slider — prediction and SHAP chart update within ~300ms
- Toggle Overtime — result changes
- No submit button visible

- [ ] **Step 3: Commit**

```bash
git add frontend/app/whatif/
git commit -m "feat: add What-If Explorer page with live debounced predictions"
```

---

## Task 15: Global Importance Page

**Files:**
- Create: `frontend/app/global/page.tsx`

- [ ] **Step 1: Create frontend/app/global/page.tsx**

```typescript
'use client'
import { useState, useEffect } from 'react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import { getGlobalImportance, getModelMetrics } from '@/lib/api'
import { GlobalImportanceResponse, ModelMetricsResponse } from '@/lib/types'
import ConfusionMatrix from '@/components/ConfusionMatrix'

export default function GlobalPage() {
  const [importance, setImportance] = useState<GlobalImportanceResponse | null>(null)
  const [metrics, setMetrics] = useState<ModelMetricsResponse | null>(null)
  const [metricsOpen, setMetricsOpen] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    Promise.all([getGlobalImportance(), getModelMetrics()])
      .then(([imp, met]) => { setImportance(imp); setMetrics(met) })
      .catch(e => setError(e.message))
  }, [])

  if (error) return <div className="text-red-600 text-sm">{error}</div>
  if (!importance) return <div className="text-slate-400 text-sm">Loading...</div>

  const chartData = importance.features.slice(0, 15).map(f => ({
    feature: f.feature,
    importance: f.importance,
  }))

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-slate-800 mb-1">Global Feature Importance</h1>
        <p className="text-slate-500 text-sm">Which features matter most across all predictions?</p>
      </div>

      {/* Bar chart */}
      <div className="bg-white rounded-lg border p-6">
        <ResponsiveContainer width="100%" height={400}>
          <BarChart
            data={chartData}
            layout="vertical"
            margin={{ top: 0, right: 20, left: 160, bottom: 0 }}
          >
            <XAxis type="number" tick={{ fontSize: 11 }} />
            <YAxis type="category" dataKey="feature" tick={{ fontSize: 12 }} width={150} />
            <Tooltip
              formatter={(val: number) => [val.toFixed(4), 'Mean |SHAP|']}
              contentStyle={{ fontSize: 12 }}
            />
            <Bar dataKey="importance" radius={2}>
              {chartData.map((_, i) => (
                <Cell key={i} fill={`hsl(${220 - i * 10}, 70%, ${55 + i * 2}%)`} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Summary */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-5">
        <p className="text-xs font-semibold text-blue-600 uppercase tracking-wide mb-2">Model Insight</p>
        <p className="text-slate-700 text-sm leading-relaxed">{importance.summary}</p>
      </div>

      {/* Collapsible metrics */}
      <div className="bg-white rounded-lg border">
        <button
          className="w-full flex items-center justify-between px-6 py-4 text-left"
          onClick={() => setMetricsOpen(o => !o)}
        >
          <span className="font-semibold text-slate-700">Model Performance</span>
          <span className="text-slate-400 text-sm">{metricsOpen ? '▲ Hide' : '▼ Show'}</span>
        </button>
        {metricsOpen && metrics && (
          <div className="px-6 pb-6 space-y-6">
            <p className="text-xs text-slate-400 italic">
              This model was trained on a sample dataset for demonstration purposes.
              In production, further validation would be required.
            </p>
            <div className="grid grid-cols-4 gap-4">
              {(['accuracy', 'precision', 'recall', 'f1'] as const).map(key => (
                <div key={key} className="bg-slate-50 rounded-lg p-4 text-center">
                  <p className="text-xs font-medium text-slate-500 uppercase tracking-wide">{key}</p>
                  <p className="text-2xl font-bold text-slate-800 mt-1">
                    {(metrics[key] * 100).toFixed(1)}%
                  </p>
                </div>
              ))}
            </div>
            <div>
              <p className="text-sm font-medium text-slate-700 mb-3">Confusion Matrix</p>
              <ConfusionMatrix matrix={metrics.confusion_matrix} />
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Test the global page manually**

Open http://localhost:3000/global.
- Verify: bar chart loads with ~15 features ranked
- Summary paragraph appears (fallback text since no API key)
- Click "Show" on Model Performance — metrics grid and confusion matrix expand
- No console errors

- [ ] **Step 3: Commit**

```bash
git add frontend/app/global/
git commit -m "feat: add Global Importance page with metrics panel"
```

---

## Task 16: README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Create README.md**

```markdown
# AttritionAI — ML Explainability Dashboard

An interactive dashboard that trains a Random Forest classifier on the IBM HR Employee Attrition dataset and makes predictions fully explainable using SHAP values. Fill in employee attributes, get a Stay/Leave prediction with confidence score, and see exactly which features drove that decision.

**Hero feature:** the What-If Explorer — adjust sliders and watch the prediction and SHAP chart update in real time.

---

## How to Run Locally

### Prerequisites
- Python 3.11+
- Node.js 18+

### Backend

```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate | macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python train.py          # trains model, takes ~30s, only needed once
uvicorn main:app --port 8000 --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:3000.

---

## Environment Variables

Copy `.env.example` to `.env` and fill in:

```
ANTHROPIC_API_KEY=your_key_here   # optional — enables AI-generated feature summary
```

---

## Technical Decisions

**Why Random Forest?** Interpretable tree structure makes SHAP fast and exact (TreeExplainer). Handles mixed categorical/numeric data without scaling. Strong baseline performance on tabular data.

**Why SHAP?** SHAP values are game-theoretically grounded — they fairly attribute each feature's contribution to the prediction. TreeExplainer makes this fast even at inference time.

**Why this dataset?** The IBM HR Attrition dataset is clean, well-understood, and poses a real business question ("Will this employee leave?") that any interviewer can immediately grasp — no domain expertise required to evaluate the output.

---

## What I'd Do Differently With More Time

- Add a proper test suite for the frontend (React Testing Library)
- Replace in-memory Claude summary cache with a file-based cache that survives restarts
- Add authentication to the `/train` endpoint
- Show partial dependence plots alongside SHAP for global explanations
- Improve mobile layout (currently desktop-only)
```

Save as `proj5/README.md`.

- [ ] **Step 2: Final smoke test**

- [ ] Backend all 4 pytest tests pass: `cd backend && pytest test_api.py -v`
- [ ] Predict page: form → result → SHAP chart renders
- [ ] What-If: sliders update result in real time
- [ ] Global Importance: bar chart + summary loads
- [ ] Model Performance: expand panel → confusion matrix renders
- [ ] No console errors in browser

- [ ] **Step 3: Final commit**

```bash
git add README.md
git commit -m "docs: add README with setup, deployment, and technical decisions"
```
