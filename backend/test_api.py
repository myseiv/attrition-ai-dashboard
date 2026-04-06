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
