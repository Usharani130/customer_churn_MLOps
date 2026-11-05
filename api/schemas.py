# Python script placeholder
# api/schemas.py
from typing import Literal
from pydantic import BaseModel, Field

class CustomerInput(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(..., ge=0, le=120, description="Months customer stayed with the company")
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["No phone service", "No", "Yes"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["No internet service", "No", "Yes"]
    OnlineBackup: Literal["No internet service", "No", "Yes"]
    DeviceProtection: Literal["No internet service", "No", "Yes"]
    TechSupport: Literal["No internet service", "No", "Yes"]
    StreamingTV: Literal["No internet service", "No", "Yes"]
    StreamingMovies: Literal["No internet service", "No", "Yes"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
    MonthlyCharges: float = Field(..., ge=0.0, description="Monthly amount charged to customer")
    TotalCharges: float = Field(..., ge=0.0, description="Total amount charged over tenure")

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }

class PredictionResponse(BaseModel):
    churn_prediction: int = Field(..., description="0 = Retain, 1 = Churn")
    churn_status: str = Field(..., description="'Yes' if churn predicted, 'No' otherwise")
    churn_probability: float = Field(..., ge=0.0, le=1.0, description="Predicted probability of churn")

class ModelInfoResponse(BaseModel):
    model_name: str
    pipeline_type: str
    expected_features: list[str]
    status: str