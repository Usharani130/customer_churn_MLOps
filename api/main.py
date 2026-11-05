# Python script placeholder
# api/main.py
import os
import joblib
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from api.schemas import CustomerInput, PredictionResponse, ModelInfoResponse

# Global artifact container
model_artifact = {
    "pipeline": None,
    "path": os.getenv("MODEL_PATH", "models/pipeline.joblib")
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifact into memory upon startup."""
    model_path = model_artifact["path"]
    if os.path.exists(model_path):
        model_artifact["pipeline"] = joblib.load(model_path)
    else:
        # App will boot, but prediction endpoints will return 503 until model is available
        model_artifact["pipeline"] = None
    yield
    model_artifact.clear()

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Production-grade inference API for Telco Customer Churn using FastAPI and Scikit-learn.",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/", tags=["General"])
async def root():
    return {
        "message": "Welcome to the Customer Churn Prediction API",
        "documentation": "/docs",
        "health_check": "/health"
    }

@app.get("/health", tags=["Monitoring"])
async def health_check():
    is_model_loaded = model_artifact["pipeline"] is not None
    return {
        "status": "healthy" if is_model_loaded else "degraded",
        "model_loaded": is_model_loaded
    }

@app.get("/model-info", response_model=ModelInfoResponse, tags=["Metadata"])
async def model_info():
    pipeline = model_artifact["pipeline"]
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded or missing from artifact store."
        )
    
    classifier_name = type(pipeline.named_steps.get("classifier", pipeline[-1])).__name__
    feature_fields = list(CustomerInput.model_fields.keys())

    return ModelInfoResponse(
        model_name=classifier_name,
        pipeline_type=type(pipeline).__name__,
        expected_features=feature_fields,
        status="ready"
    )

@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"]
)
async def predict_churn(customer: CustomerInput):
    pipeline = model_artifact["pipeline"]
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference model is currently unavailable."
        )

    try:
        input_data = pd.DataFrame([customer.model_dump()])
        prediction = int(pipeline.predict(input_data)[0])
        probability = float(pipeline.predict_proba(input_data)[0][1])

        return PredictionResponse(
            churn_prediction=prediction,
            churn_status="Yes" if prediction == 1 else "No",
            churn_probability=round(probability, 4)
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(exc)}"
        )