# Python script placeholder
# frontend/app.py
import os
import requests
import streamlit as st

# Configure page metadata and layout
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# FastAPI endpoint resolution (supports local running and container networking)
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")
PREDICT_ENDPOINT = f"{FASTAPI_URL}/predict"
HEALTH_ENDPOINT = f"{FASTAPI_URL}/health"

st.title("Telco Customer Churn Prediction")
st.markdown(
    "Enter customer demographic, subscription, and account details below to evaluate the churn risk score."
)

# Sidebar: Service connectivity monitor
st.sidebar.header("API Health Status")
try:
    health_resp = requests.get(HEALTH_ENDPOINT, timeout=3)
    if health_resp.status_code == 200 and health_resp.json().get("status") == "healthy":
        st.sidebar.success("FastAPI Service: Online")
    else:
        st.sidebar.warning("FastAPI Service: Degraded (Model missing)")
except requests.exceptions.RequestException:
    st.sidebar.error("FastAPI Service: Offline")

# Primary input form
with st.form("churn_input_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Demographics")
        gender = st.selectbox("Gender", options=["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        partner = st.selectbox("Has Partner", options=["Yes", "No"])
        dependents = st.selectbox("Has Dependents", options=["Yes", "No"])

    with col2:
        st.subheader("Subscribed Services")
        phone_service = st.selectbox("Phone Service", options=["Yes", "No"])
        multiple_lines = st.selectbox(
            "Multiple Lines",
            options=["No", "Yes", "No phone service"] if phone_service == "Yes" else ["No phone service"]
        )
        internet_service = st.selectbox("Internet Service", options=["Fiber optic", "DSL", "No"])
        
        has_internet = internet_service != "No"
        na_internet = "No internet service"

        online_security = st.selectbox("Online Security", options=["No", "Yes", na_internet] if has_internet else [na_internet])
        online_backup = st.selectbox("Online Backup", options=["No", "Yes", na_internet] if has_internet else [na_internet])
        device_protection = st.selectbox("Device Protection", options=["No", "Yes", na_internet] if has_internet else [na_internet])
        tech_support = st.selectbox("Tech Support", options=["No", "Yes", na_internet] if has_internet else [na_internet])
        streaming_tv = st.selectbox("Streaming TV", options=["No", "Yes", na_internet] if has_internet else [na_internet])
        streaming_movies = st.selectbox("Streaming Movies", options=["No", "Yes", na_internet] if has_internet else [na_internet])

    with col3:
        st.subheader("Account & Billing")
        tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12, step=1)
        contract = st.selectbox("Contract Type", options=["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", options=["Yes", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            options=[
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )
        monthly_charges = st.number_input(
            "Monthly Charges ($)",
            min_value=18.0,
            max_value=130.0,
            value=65.0,
            step=0.5
        )
        # Automatic sensible default estimation for TotalCharges based on tenure
        suggested_total = round(float(tenure * monthly_charges), 2)
        total_charges = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            max_value=10000.0,
            value=max(suggested_total, 18.0) if tenure > 0 else 0.0,
            step=1.0
        )

    submit_button = st.form_submit_button(label="Predict Churn Risk", use_container_width=True)

# Inference execution & response handling
if submit_button:
    payload = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": int(tenure),
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": float(monthly_charges),
        "TotalCharges": float(total_charges)
    }

    with st.spinner("Connecting to inference engine..."):
        try:
            response = requests.post(PREDICT_ENDPOINT, json=payload, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                is_churn = result.get("churn_prediction") == 1
                probability = result.get("churn_probability", 0.0)

                st.divider()
                st.subheader("Prediction Results")

                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    if is_churn:
                        st.error("Churn Status: HIGH RISK (Likely to leave)")
                    else:
                        st.success("Churn Status: LOW RISK (Likely to stay)")

                with res_col2:
                    st.metric(
                        label="Churn Probability",
                        value=f"{probability * 100:.2f}%",
                        delta=f"{(probability - 0.5) * 100:.1f}% vs threshold" if is_churn else None,
                        delta_color="inverse"
                    )

                # Progress gauge
                st.progress(probability)
                
            elif response.status_code == 422:
                st.error("Validation Error (422): Provided input data format does not match API schema.")
                st.json(response.json())
            elif response.status_code == 503:
                st.warning("Inference Unavailable (503): Model artifact not loaded on the backend server.")
            else:
                st.error(f"Inference server error ({response.status_code}): {response.text}")

        except requests.exceptions.ConnectionError:
            st.error(
                f"Connection Refused: Failed to reach FastAPI backend at `{FASTAPI_URL}`. "
                "Ensure the Uvicorn service is running (`uvicorn api.main:app --port 8000`)."
            )
        except requests.exceptions.Timeout:
            st.error("Request Timed Out: FastAPI service took too long to return a prediction.")
        except Exception as ex:
            st.error(f"An unexpected error occurred: {str(ex)}")