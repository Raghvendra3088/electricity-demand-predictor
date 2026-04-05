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
    # Organize inputs exactly in the order the model expects, as a 2D Numpy array
    # This bypasses strict Pandas column-name validation
    input_array = np.array([[hour, day, month, sub_1, sub_2, sub_3]])
    
    try:
        # Scale the inputs
        scaled_data = scaler.transform(input_array)
        
        # Predict
        prediction = model.predict(scaled_data)[0]
        
        st.success(f"Predicted Global Intensity: **{prediction:.2f} Amps**")
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
