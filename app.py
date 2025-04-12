import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# Load model and feature names
model = joblib.load("model.pkl")
with open("features.json", "r") as f:
    feature_names = json.load(f)

st.set_page_config(page_title="Loan Approval Predictor", layout="centered")
st.title("🏦 Loan Approval Prediction App")
st.markdown("Enter applicant details to predict whether the loan will be **approved** or **rejected**.")

# --- Sidebar Inputs ---
st.sidebar.header("Applicant Information")

income_annum = st.sidebar.number_input("Annual Income (₹)", min_value=100000, max_value=10000000, step=10000)
loan_amount = st.sidebar.number_input("Requested Loan Amount (₹)", min_value=50000, max_value=50000000, step=10000)
loan_term = st.sidebar.number_input("Loan Term (months)", min_value=6, max_value=360, step=6)
cibil_score = st.sidebar.slider("CIBIL Score", 300, 900, 650)
no_of_dependents = st.sidebar.slider("Number of Dependents", 0, 5, 0)

education = st.sidebar.selectbox("Education", ["graduate", "not graduate"])
self_employed = st.sidebar.selectbox("Self Employed", ["no", "yes"])

res_assets = st.sidebar.number_input("Residential Assets Value (₹)", min_value=0)
comm_assets = st.sidebar.number_input("Commercial Assets Value (₹)", min_value=0)
lux_assets = st.sidebar.number_input("Luxury Assets Value (₹)", min_value=0)
bank_assets = st.sidebar.number_input("Bank Assets Value (₹)", min_value=0)

# --- Feature Engineering ---
total_assets = res_assets + comm_assets + lux_assets + bank_assets
credit_to_assets_ratio = loan_amount / total_assets if total_assets > 0 else 0
log_loan_amount = np.log1p(loan_amount)
log_income_annum = np.log1p(income_annum)

# --- Build input vector ---
input_dict = {
    "loan_term": loan_term,
    "cibil_score": cibil_score,
    "no_of_dependents": no_of_dependents,
    "residential_assets_value": res_assets,
    "commercial_assets_value": comm_assets,
    "luxury_assets_value": lux_assets,
    "bank_asset_value": bank_assets,
    "total_assets": total_assets,
    "credit_to_assets_ratio": credit_to_assets_ratio,
    "log_loan_amount": log_loan_amount,
    "log_income_annum": log_income_annum,
    "education_ not graduate": 1 if education == "not graduate" else 0,
    "self_employed_ yes": 1 if self_employed == "yes" else 0
}

# Convert to DataFrame with same column order as training
input_df = pd.DataFrame([input_dict])[feature_names]

# --- Predict ---
if st.button("🔍 Predict Loan Approval"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 0:
        st.success(f"✅ Loan Approved! Probability of rejection: {probability:.2%}")
    else:
        st.error(f"❌ Loan Rejected. Probability of rejection: {probability:.2%}")

    st.subheader("🔍 Model Inputs Summary")
    st.dataframe(input_df.T.rename(columns={0: "Value"}))
