# ML Explainability Dashboard — Design Spec

**Date:** 2026-04-06
**Project:** proj5 — ML Model Explainability Dashboard
**Status:** Approved

---

## Overview

An interactive web dashboard that trains a Random Forest classifier on the IBM HR Employee Attrition dataset, makes predictions on employee attributes, and explains those predictions using SHAP values. Non-technical users can fill in a form, get a prediction (Stay/Leave), see why the model made that decision, and experiment with what-if scenarios via live sliders.

---

## Decisions Made

| Decision | Choice | Reason |
|---|---|---|
| Dataset delivery | Bundled CSV in repo | ~300KB, no licence issues, zero setup friction for reviewers |
| Build strategy | Backend-first | ML pipeline is the most uncertain part; build and verify before adding UI |
| Layout | Top nav + two-column | Form and results visible side by side without scrolling |
| Claude API key | Stubbed (fallback text) | Key not available at build time; wire in later |
| State management | Local `useState` per page | Pages don't share state; no need for a global store |

---

## Stack

- **Backend:** Python 3.11+, FastAPI, scikit-learn, SHAP, pandas, joblib
- **Frontend:** Next.js 14 (app router), Tailwind CSS, Recharts
- **AI:** Anthropic Claude API (one call, cached; falls back to hardcoded string)

---

## File Structure

```
proj5/
├── backend/
│   ├── main.py              # FastAPI app, loads model at startup
│   ├── train.py             # Training pipeline, run once
│   ├── test_api.py          # pytest tests
│   ├── requirements.txt
│   ├── data/
│   │   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
│   └── model/               # gitignored — produced by train.py
│       ├── rf_model.pkl
│       └── shap_explainer.pkl
├── frontend/
│   ├── app/
│   │   ├── page.tsx         # Predict + Explain view
│   │   ├── whatif/
│   │   │   └── page.tsx     # What-If Explorer
│   │   └── global/
│   │       └── page.tsx     # Global Importance + Model Metrics
│   ├── components/
│   │   ├── ShapChart.tsx
│   │   ├── ConfusionMatrix.tsx
│   │   └── PredictionResult.tsx
│   ├── package.json
│   └── tailwind.config.ts
├── .env.example
└── docs/
    └── superpowers/specs/
        └── 2026-04-06-ml-explainability-dashboard-design.md
```

---

## The 10 UI Features

Selected by global SHAP importance on the IBM attrition dataset:

| Feature | Type | UI Control |
|---|---|---|
| OverTime | Categorical (Yes/No) | Dropdown |
| Age | Numeric (18–60) | Number input / slider |
| MonthlyIncome | Numeric (1000–20000) | Number input / slider |
| JobSatisfaction | Ordinal (1–4) | Dropdown |
| YearsAtCompany | Numeric (0–40) | Number input / slider |
| WorkLifeBalance | Ordinal (1–4) | Dropdown |
| JobLevel | Ordinal (1–5) | Dropdown |
| DistanceFromHome | Numeric (1–30) | Number input / slider |
| NumCompaniesWorked | Numeric (0–9) | Number input / slider |
| StockOptionLevel | Ordinal (0–3) | Dropdown |

All other dataset features are set to their median/mode values when constructing a prediction row.

---

## Backend Design

### train.py

1. Load CSV from `data/`
2. Drop constant columns: `EmployeeCount`, `Over18`, `StandardHours`, `EmployeeNumber`
3. Encode target: `Attrition` → 1 (Yes), 0 (No)
4. LabelEncode all categorical columns; store encoder mappings
5. 80/20 train/test split (random_state=42)
6. Fit `RandomForestClassifier(n_estimators=100, random_state=42)`
7. Evaluate: accuracy, precision, recall, F1, confusion matrix
8. Fit `shap.TreeExplainer(model)` on training set
9. Compute global mean absolute SHAP values across training set
10. Save `rf_model.pkl`, `shap_explainer.pkl`, `encoders.pkl`, `global_shap.json`, `metrics.json` to `model/`

### main.py

Loads all model artifacts at startup. If `model/` is missing, calls `train.py` programmatically before serving.

**`POST /predict`**
- Input: `{ "OverTime": "Yes", "Age": 32, ... }` (10 fields)
- Validate all 10 fields present; return 422 with field-level error if not
- Build full feature row: use submitted values for 10 features, median/mode for the rest
- Encode using saved encoders
- `model.predict_proba()` → confidence score for positive class (Leave)
- `explainer.shap_values(row)[1]` → local SHAP values
- Sort by absolute value, return top 10
- Response: `{ prediction, confidence, shap_values: [{ feature, value, shap, direction }] }`

**`GET /global-importance`**
- Returns precomputed `global_shap.json` (all features ranked by mean absolute SHAP)
- On first call: if `ANTHROPIC_API_KEY` is set, calls Claude API with top 5 features, caches response in memory
- If no key or API error: returns hardcoded fallback summary string
- Response: `{ features: [...], summary: "..." }`

**`GET /model-metrics`**
- Returns contents of `metrics.json`: accuracy, precision, recall, F1, confusion_matrix

**`POST /train`**
- Reruns training pipeline
- Returns new metrics

CORS: allow `http://localhost:3000`.

---

## Frontend Design

### Navigation
Top nav bar (dark background): logo left, three links right — **Predict**, **What-If**, **Global Importance**. Active link underlined.

### `/` — Predict + Explain (two-column)

**Left column — Form:**
- 10 inputs: dropdowns for categoricals/ordinals, number inputs for numerics
- Default values pre-filled (median/mode so the form works immediately on load)
- "Predict" submit button

**Right column — Results:**
- Before first submit: placeholder skeleton with "Fill in the form to get a prediction"
- After submit: `PredictionResult` component (large badge — green "STAY" or amber "LEAVE" + confidence %) then `ShapChart`

**`ShapChart`:** Horizontal bar chart via Recharts. Bars left of centre = stay-pushing (blue), bars right = leave-pushing (red). Each bar labelled with feature name and its input value. Y-axis = feature names. X-axis = SHAP value.

### `/whatif` — What-If Explorer (two-column)

Same layout as Predict page. Left column: sliders for numerics (min/max/step hardcoded per feature), dropdowns for categoricals. Right column: live `PredictionResult` + `ShapChart`. Prediction updates on every input change (debounced 300ms API call). No submit button.

### `/global` — Global Importance

Full-width `Recharts` bar chart: all features ranked by mean absolute SHAP, horizontal bars. Below: Claude summary paragraph (or fallback). Below that: collapsible "Model Performance" section containing `ConfusionMatrix`.

**`ConfusionMatrix`:** 2×2 coloured grid. Cells: TN (green), FP (red), FN (red), TP (green). Each cell shows the count and a label (True Negative, etc.).

### Loading & Error States
- All API calls show a loading spinner in the results area
- API errors show an inline error message (not a crash)
- Global importance page fetches on mount with a loading skeleton

---

## Error Handling

| Scenario | Behaviour |
|---|---|
| Model not trained at startup | Auto-runs train.py, logs to console |
| Missing field in `/predict` | 422 with field-level validation error |
| SHAP/model error | 500 with error message |
| Claude API unavailable/no key | Silent fallback to hardcoded summary |
| Frontend API error | Inline error message in results column |

---

## Testing

**`backend/test_api.py` (pytest):**
1. `POST /predict` with valid 10-field input → 200, correct response shape
2. `POST /predict` with missing fields → 422
3. `GET /model-metrics` → 200, contains accuracy, confusion_matrix keys
4. `GET /global-importance` → 200, contains features array and summary string

**Manual smoke test checklist (in README):**
- [ ] Train runs without errors
- [ ] Predict returns a result with SHAP chart
- [ ] What-If sliders update result in real time
- [ ] Global importance loads with summary
- [ ] Confusion matrix renders correctly

---

## Environment Variables

```
ANTHROPIC_API_KEY=   # optional — Claude summary falls back gracefully without it
```

---

## Not In Scope

- Authentication on any endpoint
- User accounts or saved predictions
- Model versioning or experiment tracking
- Production-hardened error handling
- Frontend unit tests
