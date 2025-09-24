import streamlit as st
import pandas as pd
import pickle

# -------------------------------
# Load model and encoders
# -------------------------------
with open("customer_churn_model.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
feature_names = data["features_names"]

# Load encoders (saved separately during training)
with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

st.title("📊 Customer Churn Prediction App")
st.write("Fill in the customer details below to predict churn.")

# -------------------------------
# Input Form
# -------------------------------
with st.form("churn_form"):
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (months)", min_value=0, step=1)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, step=1.0)
    total_charges = st.number_input("Total Charges", min_value=0.0, step=1.0)

    submitted = st.form_submit_button("Predict")

# -------------------------------
# Prediction
# -------------------------------
if submitted:
    # Build dataframe from user input
    input_data = {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }

    input_df = pd.DataFrame([input_data])

    # Apply encoders
    for column, encoder in encoders.items():
        if column in input_df.columns:
            input_df[column] = encoder.transform(input_df[column])

    # Reorder columns to match training
    input_df = input_df[feature_names]

    # Make prediction
    prediction = model.predict(input_df)[0]
    pred_prob = model.predict_proba(input_df)[0]

    # Display results
    if prediction == 1:
        st.error(f"🚨 Prediction: **Churn** (Probability: {pred_prob[1]:.2f})")
    else:
        st.success(f"✅ Prediction: **No Churn** (Probability: {pred_prob[0]:.2f})")
