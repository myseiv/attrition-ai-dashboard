# PRD — Project 5: ML Model Explainability Dashboard

> **START HERE. Build this project first.**

---

## What It Is

An interactive web dashboard that trains a machine learning classifier on a real dataset, then makes its predictions fully explainable to a non-technical user. The user can input values, get a prediction, see why the model made that decision, and experiment with what-if scenarios via interactive sliders.

This is not an LLM wrapper. It involves real model training, real ML explainability techniques (SHAP), and a UI that makes those outputs legible. That combination — ML depth + design quality — is rare in portfolios and memorable in interviews.

---

## The Problem It Solves

Most ML models are black boxes. Practitioners train them, get an accuracy number, and ship them. This project demonstrates the opposite: that good engineering includes making model behaviour understandable, auditable, and interactive. This is directly relevant to AI safety, product teams evaluating models, and any company using ML in a production decision pipeline.

---

## Dataset

Use the **IBM HR Employee Attrition dataset** (publicly available, clean, well-understood):
- Source: Kaggle / IBM sample data
- Target variable: `Attrition` (Yes/No — will the employee leave?)
- ~1,470 rows, ~35 features
- No licence issues, no scraping required

Why this dataset: the prediction task is immediately understandable to any interviewer. "Will this employee leave?" is a real business question with real stakes — it grounds the technical work in something concrete.

---

## Core Features

### 1. Model training pipeline (backend)
- Train a Random Forest classifier on the attrition dataset
- Perform basic feature engineering (encode categoricals, normalise numerics)
- Evaluate the model: accuracy, precision, recall, F1, confusion matrix
- Compute global SHAP values across the training set
- Persist the trained model and SHAP explainer to disk (pickle or joblib)
- This runs once at startup or via a `/train` endpoint — not on every request

### 2. Prediction interface (frontend)
- A clean form where the user fills in employee attributes (use the 10 most important features — not all 35, that's too many for a UI)
- On submit: call the backend, get back a prediction (Stay / Leave) with a confidence score
- Display the result clearly — not just a number, a human-readable verdict

### 3. Local SHAP explanation
- For each prediction, compute a local SHAP explanation (why did the model make this specific decision?)
- Display as a horizontal bar chart: features that pushed toward "Leave" on one side, features that pushed toward "Stay" on the other
- Label each bar with the feature name and its value — the user should be able to read "High overtime hours increased leave probability by 23%"

### 4. Global feature importance
- A separate view showing which features matter most across all predictions globally
- Bar chart ranked by mean absolute SHAP value
- Brief AI-generated summary (Claude API): "The three biggest predictors of attrition in this dataset are..." — call Claude with the top features and ask for a plain English interpretation

### 5. What-if explorer
- Sliders and dropdowns for each of the 10 key features
- As the user adjusts a slider, the prediction and SHAP chart update in real time (debounced API calls)
- This is the most impressive part — let the user see the model respond to changes live

### 6. Model performance panel
- A collapsible section showing the model's evaluation metrics
- Confusion matrix (rendered as a heatmap, not a table)
- Honest framing: "This model was trained on a sample dataset for demonstration purposes. In production, further validation would be required."

---

## Implementation

### Stack
- **Backend**: Python, FastAPI
- **ML**: scikit-learn (RandomForestClassifier), SHAP, pandas, joblib
- **Frontend**: React (Next.js), Tailwind CSS, Recharts for charts
- **AI**: Claude API for the global feature summary paragraph only

### File structure
```
/backend
  main.py          — FastAPI app
  train.py         — training pipeline, run once
  model/           — persisted model + explainer
  data/            — raw dataset
/frontend
  app/
    page.tsx       — prediction form + results
    global/        — global importance view
    whatif/        — what-if explorer
  components/
    ShapChart.tsx
    ConfusionMatrix.tsx
    PredictionResult.tsx
```

### Key API endpoints
- `POST /predict` — takes feature values, returns prediction + confidence + local SHAP values
- `GET /global-importance` — returns global SHAP values + Claude-generated summary
- `GET /model-metrics` — returns accuracy, F1, confusion matrix
- `POST /train` — (admin only) retrains the model

### SHAP implementation note
Use `shap.TreeExplainer` with the trained RandomForest. Compute `shap_values[1]` for the positive class (Leave). For local explanations, return the top 10 features by absolute SHAP value for that specific prediction. Do not return all features — it overwhelms the UI.

### Claude API usage
One call only: when the global importance view loads, send the top 5 features and their mean SHAP values to Claude with this system prompt:

```
You are a data analyst explaining a machine learning model to a business stakeholder.
Given the following feature importances from an employee attrition model, write 2-3 sentences
in plain English explaining what the model has learned. Be specific. Do not use jargon.
Do not mention SHAP or Random Forest. Focus on what the business should pay attention to.
```

Cache this response — do not call Claude on every page load.

---

## UI Design Direction

Clean, professional, data-dense but not cluttered. Think Retool or Linear, not a Jupyter notebook.

- Dark sidebar for navigation between views
- White/light grey main content area
- Prediction result displayed as a large, clear badge: green for Stay, amber for Leave
- SHAP chart: horizontal bars, blue for Stay-pushing features, red for Leave-pushing features
- Sliders in the what-if explorer should feel responsive and immediate

Do not use a component library with heavy defaults (no Material UI, no Ant Design). Tailwind + Recharts is sufficient and keeps the design clean.

---

## What to Show in the Portfolio

The what-if explorer is the hero feature — lead with it in your README GIF/screenshot. It shows the model is live and interactive, not a static demo.

Second highlight: the SHAP chart with labelled feature contributions. This is what makes the project technically credible to an ML practitioner.

Third: the Claude-generated summary. It shows you know how to use AI as one tool in a larger system, not the entire system.

---

## Deployment

- Backend: Railway or Hugging Face Spaces (Python)
- Frontend: Vercel
- The trained model is committed to the repo (it's small — Random Forest on 1,470 rows is ~2MB)
- Environment variable required: `ANTHROPIC_API_KEY`

---

## README Must Include

- What the project does (1 paragraph)
- Screenshot or GIF of the what-if explorer
- How to run locally (step by step)
- How to deploy
- Technical decisions: why Random Forest, why SHAP, why this dataset
- What you would do differently with more time (honest reflection — interviewers respect this)

---

## Definition of Done

- [ ] Model trains without errors on the IBM attrition dataset
- [ ] Prediction form returns a result with confidence score
- [ ] SHAP chart renders correctly for each prediction
- [ ] What-if sliders update prediction in real time
- [ ] Global importance view loads with Claude-generated summary
- [ ] Model metrics panel shows confusion matrix heatmap
- [ ] Deployed to a public URL
- [ ] README complete
