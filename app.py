import streamlit as st
import pandas as pd
import numpy as np
import joblib
try:
    model = joblib.load("loan_approval_model.pkl")
    print("done")
except FileNotFoundError:
    print("not found")
    exit()

st.title("Loan Approval Prediction App")
st.image("https://static.vecteezy.com/system/resources/previews/015/397/805/non_2x/businessman-check-and-check-mark-online-business-agreements-and-approvals-business-contract-signing-confirmation-of-contract-documents-or-warranty-card-idea-of-employment-project-review-free-photo.jpg")
st.sidebar.header("Applicant Information")

no_of_dependents = st.sidebar.number_input("Number of Dependents", min_value=0, step=1)
education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
income_annum = st.sidebar.number_input("Annual Income", min_value=0)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0)
loan_term = st.sidebar.number_input("Loan Term (months)", min_value=0)
cibil_score = st.sidebar.slider("CIBIL Score", min_value=300, max_value=900)
residential_assets_value = st.sidebar.number_input("Residential Assets Value", min_value=0)
commercial_assets_value = st.sidebar.number_input("Commercial Assets Value", min_value=0)
luxury_assets_value = st.sidebar.number_input("Luxury Assets Value", min_value=0)
bank_asset_value = st.sidebar.number_input("Bank Asset Value", min_value=0)

def preprocess_input():
    return np.array([
        no_of_dependents,
        1 if education == "Graduate" else 0,
        1 if self_employed == "Yes" else 0,
        income_annum,
        loan_amount,
        loan_term,
        cibil_score,
        residential_assets_value,
        commercial_assets_value,
        luxury_assets_value,
        bank_asset_value
    ]).reshape(1, -1)

if st.sidebar.button("Predict"):
    input_data = preprocess_input()
    prediction = model.predict(input_data)
    result = "Approved" if prediction[0] == 1 else "Rejected"
    st.subheader("Prediction Result")
    st.success(result)