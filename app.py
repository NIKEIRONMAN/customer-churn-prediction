import streamlit as st
import pandas as pd
import pickle

# Load trained model
with open("churn_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Customer Churn Prediction App")

# Input fields
tenure = st.number_input("Tenure (months)", min_value=0)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0)
total_charges = st.number_input("Total Charges", min_value=0.0)

# Predict button
if st.button("Predict"):
    input_data = pd.DataFrame([[tenure, monthly_charges, total_charges]],
                              columns=["tenure", "MonthlyCharges", "TotalCharges"])
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error(" Customer is likely to churn.")
    else:
        st.success(" Customer is likely to stay.")
