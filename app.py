import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the saved model and scaler
model = joblib.load('rf_model.joblib')
scaler = joblib.load('scaler.joblib')

st.title("⚡ Electricity Demand Predictor")
st.write("Enter the time and sub-metering details below to predict the global intensity (Amps).")

# Create input fields for the user
col1, col2, col3 = st.columns(3)

with col1:
    hour = st.slider("Hour of Day (0-23)", 0, 23, 12)
    sub_1 = st.number_input("Kitchen Active Energy (Wh)", value=0.0)

with col2:
    day = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 0)
    sub_2 = st.number_input("Laundry Active Energy (Wh)", value=0.0)

with col3:
    month = st.slider("Month (1-12)", 1, 12, 6)
    sub_3 = st.number_input("Water Heater/AC Energy (Wh)", value=0.0)

# Prediction Button
if st.button("Predict Demand"):
    # Organize inputs exactly how the model expects them
    input_data = pd.DataFrame({
        'hour': [hour],
        'day': [day],
        'month': [month],
        'Sub_metering_1': [sub_1],
        'Sub_metering_2': [sub_2],
        'Sub_metering_3': [sub_3]
    })
    
    # Scale the inputs
    scaled_data = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(scaled_data)[0]
    
    st.success(f"Predicted Global Intensity: **{prediction:.2f} Amps**")