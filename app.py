# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import xgboost as xgb
import os

st.set_page_config(page_title="Churn Predictor", layout="centered")

st.title("Churn Predictor")
st.write("Fill the customer fields below and click Predict. The app uses your saved preprocessor and local XGBoost model.")

# --- Load artifacts (preprocessor, raw column order, model) ---
@st.cache_resource
def load_artifacts():
    # preprocessor
    preproc_path = "preprocessor.joblib"
    raw_cols_path = "preprocessor_raw_columns.json"
    model_path = os.path.join("xgboost-model")
    missing = []
    if not os.path.exists(preproc_path):
        missing.append(preproc_path)
    if not os.path.exists(raw_cols_path):
        missing.append(raw_cols_path)
    if not os.path.exists(model_path):
        missing.append(model_path)
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}. Place them next to app.py.")
    preprocessor = joblib.load(preproc_path)
    with open(raw_cols_path, "r") as f:
        raw_cols = json.load(f)
    model = xgb.Booster()
    model.load_model(model_path)
    return preprocessor, raw_cols, model

try:
    preprocessor, RAW_COLS, model = load_artifacts()
except Exception as e:
    st.error(f"Error loading artifacts: {e}")
    st.stop()

# --- Helper: build input form ---
st.sidebar.header("Threshold & Controls")
threshold = st.sidebar.slider("Decision threshold (probability >= threshold => Churn)", 0.0, 1.0, 0.5, 0.01)
st.sidebar.caption("Lower threshold => higher recall (catch more churners).")

st.header("Customer details")

# Categorical options (based on dataset)
gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("Senior Citizen (0 = No, 1 = Yes)", [0, 1])
Partner = st.selectbox("Partner", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12, step=1)
PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
InternetService = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
PaymentMethod = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, max_value=10000.0, value=70.0, step=0.1, format="%.2f")
TotalCharges = st.number_input("Total Charges", min_value=0.0, max_value=1e7, value=840.0, step=0.1, format="%.2f")

# Build raw input dict in same keys as original raw columns
input_dict = {
    "gender": gender,
    "SeniorCitizen": SeniorCitizen,
    "Partner": Partner,
    "Dependents": Dependents,
    "tenure": tenure,
    "PhoneService": PhoneService,
    "MultipleLines": MultipleLines,
    "InternetService": InternetService,
    "OnlineSecurity": OnlineSecurity,
    "OnlineBackup": OnlineBackup,
    "DeviceProtection": DeviceProtection,
    "TechSupport": TechSupport,
    "StreamingTV": StreamingTV,
    "StreamingMovies": StreamingMovies,
    "Contract": Contract,
    "PaperlessBilling": PaperlessBilling,
    "PaymentMethod": PaymentMethod,
    "MonthlyCharges": MonthlyCharges,
    "TotalCharges": TotalCharges
}

st.subheader("Preview input")
st.json(input_dict)

# Predict button
if st.button("Predict churn probability"):
    try:
        # Create DataFrame with exact raw column order
        df_new = pd.DataFrame([input_dict])
        # Reindex to RAW_COLS; if user misses a column, it will be filled with NaN
        df_new = df_new.reindex(columns=RAW_COLS)
        # Optional: fill NaN with sensible defaults (here: numeric 0, categorical -> most common approach)
        # For safety, fill numeric NaN with 0 and string NaN with "No"
        for c in df_new.columns:
            if pd.api.types.is_numeric_dtype(df_new[c]):
                df_new[c] = df_new[c].fillna(0)
            else:
                df_new[c] = df_new[c].fillna("No")
        # Transform
        X_new = preprocessor.transform(df_new)
        X_new_arr = np.asarray(X_new)
        dnew = xgb.DMatrix(X_new_arr)
        prob = float(model.predict(dnew)[0])
        pred_label = "Churn" if prob >= threshold else "No Churn"

        st.success(f"Probability of churn: {prob:.3f}")
        st.info(f"Prediction (threshold={threshold:.2f}): {pred_label}")

        # show transformed features optionally
        if st.checkbox("Show preprocessed features (first 10 values)"):
            st.write("Shape:", X_new_arr.shape)
            # convert to 1D list
            vals = X_new_arr.flatten().tolist()
            st.write(vals[:50])  # show first 50 features
    except Exception as ex:
        st.error(f"Prediction failed: {ex}")
