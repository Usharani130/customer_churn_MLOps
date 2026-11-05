# Python script placeholder
from fastapi.testclient import TestClient
from api.main import app, model_artifact

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "documentation" in response.json()

def test_health_check_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
    assert "model_loaded" in response.json()

def test_predict_success(monkeypatch, mock_trained_pipeline, sample_valid_customer_payload):
    # Inject mock pipeline into application state to bypass missing file on disk
    monkeypatch.setitem(model_artifact, "pipeline", mock_trained_pipeline)
    
    response = client.post("/predict", json=sample_valid_customer_payload)
    assert response.status_code == 200
    
    data = response.json()
    assert data["churn_prediction"] in [0, 1]
    assert data["churn_status"] in ["Yes", "No"]
    assert 0.0 <= data["churn_probability"] <= 1.0

def test_predict_validation_error(sample_invalid_customer_payload):
    response = client.post("/predict", json=sample_invalid_customer_payload)
    assert response.status_code == 422  # Unprocessable Entity