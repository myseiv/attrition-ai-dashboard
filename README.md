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
