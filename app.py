import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("🏋️ Membership Cancellation Predictor")

# Input widgets
duration = st.slider("Membership Duration (days)", 0, 3650, 180)
gender = st.selectbox("Gender", ["Male", "Female"])
marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
age = st.slider("Age at Membership Issue", 18, 75, 30)
occupation = st.selectbox("Occupation Code", [0, 1, 2, 3, 4, 5])
income = st.number_input("Annual Income", value=50000)
fees = st.number_input("Annual Fees", value=1200)
additional = st.slider("Additional Members", 0, 5, 1)
package = st.selectbox("Package Type", ["Type-A", "Type-B"])
payment = st.selectbox("Payment Mode", ["Monthly", "Quarterly", "Annual"])

# Manual encoding (match your Kaggle encoders!)
gender_map = {"Male": 0, "Female": 1}
marital_map = {"Single": 0, "Married": 1, "Divorced": 2, "Widowed": 3}
package_map = {"Type-A": 0, "Type-B": 1}
payment_map = {"Monthly": 0, "Quarterly": 1, "Annual": 2}

# Prepare input
input_data = np.array([[
    duration,
    gender_map[gender],
    marital_map[marital],
    age,
    occupation,
    income,
    package_map[package],
    fees,
    additional,
    payment_map[payment]
]])
input_scaled = scaler.transform(input_data)

# Predict
if st.button("Predict"):
    result = model.predict(input_scaled)[0]
    if result == 1:
        st.success("✅ The member is likely to stay (INFORCE)")
    else:
        st.error("❌ The member is likely to cancel (CANCELLED)")
