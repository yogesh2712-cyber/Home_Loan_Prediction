import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Page Config
st.set_page_config(page_title="Loan Prediction App", layout="wide")

st.title("🏦 Loan Eligibility Prediction System")
st.markdown("Predict whether a loan will be approved based on applicant details.")

# Load Model & Features
@st.cache_resource
def load_model():
    model = pickle.load(open('loan_model.pkl', 'rb'))
    features = pickle.load(open('model_features.pkl', 'rb'))
    return model, features

model, model_features = load_model()

# Sidebar Input

st.sidebar.header("📋 Enter Applicant Details")

# Numeric Inputs
income = st.sidebar.number_input("💰 Applicant Income", min_value=0)
loan_amount = st.sidebar.number_input("🏦 Loan Amount", min_value=0)
credit_score = st.sidebar.number_input("📊 Credit Score", min_value=300, max_value=900)

# Categorical Inputs
gender = st.sidebar.selectbox("👤 Gender", ["Male", "Female"])
married = st.sidebar.selectbox("💍 Married", ["Yes", "No"])
education = st.sidebar.selectbox("🎓 Education", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("💼 Self Employed", ["Yes", "No"])

# Convert Inputs to DataFrame

input_dict = {
    "income": income,
    "loan_amount": loan_amount,
    "credit_score": credit_score,
    "gender": gender,
    "married": married,
    "education": education,
    "self_employed": self_employed
}

input_df = pd.DataFrame([input_dict])

# Encoding (IMPORTANT)
# ----------------------------
# Convert categorical → numeric (same as training)
input_df = pd.get_dummies(input_df)

# Align with model features
input_df = input_df.reindex(columns=model_features, fill_value=0)

# ----------------------------
# Prediction
# ----------------------------
if st.button("🔍 Predict Loan Status"):

    try:
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.subheader("📊 Prediction Result")

        if prediction == 1:
            st.success("✅ Loan Approved")
        else:
            st.error("❌ Loan Rejected")

        st.write(f"📈 Approval Probability: **{probability:.2f}**")

        # Show input summary
        st.subheader("📄 Input Summary")
        st.dataframe(input_df)

    except Exception as e:
        st.error(f"Error: {e}")
