# Python script placeholder
import pandas as pd

def test_mock_pipeline_prediction(mock_trained_pipeline):
    """Ensure pipeline produces expected binary output and bounded probabilities."""
    sample = pd.DataFrame([{
        "tenure": 2,
        "MonthlyCharges": 85.0,
        "TotalCharges": 170.0
    }])
    
    prediction = mock_trained_pipeline.predict(sample)
    proba = mock_trained_pipeline.predict_proba(sample)
    
    assert prediction[0] in [0, 1]
    assert 0.0 <= proba[0][1] <= 1.0
    assert proba.shape == (1, 2)