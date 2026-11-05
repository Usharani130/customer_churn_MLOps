# Python script placeholder
import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer

@pytest.fixture
def sample_valid_customer_payload():
    """Valid customer JSON payload matching api/schemas.py."""
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35,
        "TotalCharges": 844.20
    }

@pytest.fixture
def sample_invalid_customer_payload():
    """Payload violating schema boundaries and categorical bounds."""
    return {
        "gender": "UnknownGender",  # Invalid literal
        "SeniorCitizen": 5,         # Invalid literal (must be 0 or 1)
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": -5,               # Violates ge=0
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Satellite", # Invalid literal
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Hourly",       # Invalid literal
        "PaperlessBilling": "Maybe",
        "PaymentMethod": "Bitcoin",
        "MonthlyCharges": -10.0,    # Violates ge=0.0
        "TotalCharges": 100.0
    }

@pytest.fixture
def mock_trained_pipeline():
    """Generates an in-memory Scikit-learn pipeline for deterministic unit tests."""
    X = pd.DataFrame({
        "tenure": [1, 24, 60],
        "MonthlyCharges": [70.0, 20.0, 110.0],
        "TotalCharges": [70.0, 480.0, 6600.0]
    })
    y = np.array([1, 0, 0])
    
    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), ["tenure", "MonthlyCharges", "TotalCharges"])],
        remainder="drop"
    )
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(random_state=42))
    ])
    pipeline.fit(X, y)
    return pipeline