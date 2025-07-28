import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

# Load trained model
model = load("air_quality_multi_model.pkl")

# App title
st.set_page_config(page_title="Air Quality Predictor", layout="centered")
st.title("🌍 Air Quality Prediction App")
st.markdown("Enter the sensor values below to predict air pollutant levels and get an air quality assessment.")

# Sidebar input fields
st.sidebar.header("Sensor Inputs")
t = st.sidebar.number_input("Temperature (°C)", min_value=-50.0, max_value=100.0, value=25.0)
rh = st.sidebar.number_input("Relative Humidity (%)", min_value=0.0, max_value=100.0, value=40.0)
ah = st.sidebar.number_input("Absolute Humidity", min_value=0.0, max_value=50.0, value=1.0)
s1 = st.sidebar.number_input("Sensor 1 (PT08.S1(CO))", min_value=0.0, max_value=10000.0, value=1200.0)
s5 = st.sidebar.number_input("Sensor 5 (PT08.S5(O3))", min_value=0.0, max_value=10000.0, value=1000.0)

# Prediction button
if st.sidebar.button("Predict Air Quality"):
    # Create input DataFrame
    input_data = pd.DataFrame([{
        "T": t,
        "RH": rh,
        "AH": ah,
        "PT08.S1(CO)": s1,
        "PT08.S5(O3)": s5
    }])

    try:
        # Model prediction
        predictions = model.predict(input_data)[0]
        components = ['CO(GT)', 'NOx(GT)', 'NO2(GT)', 'C6H6(GT)']

        st.subheader("🔬 Predicted Air Quality Components")
        total_score = 0

        for name, value in zip(components, predictions):
            st.write(f"**{name}**: {value:.2f} mg/m³")
            total_score += value

        st.markdown("---")
        st.success(f"**Total Air Quality Score:** {total_score:.2f} mg/m³")

        # Quality assessment
        if total_score < 50:
            quality = "✅ Good – Air quality is satisfactory. 😊"
            color = "green"
        elif total_score < 100:
            quality = "🟡 Moderate – Acceptable, but may affect sensitive groups. 😐"
            color = "orange"
        elif total_score < 150:
            quality = "⚠️ Unhealthy for Sensitive Groups – Limit exposure. 😷"
            color = "red"
        else:
            quality = "❌ Unhealthy – Health effects for everyone. 😫"
            color = "darkred"

        st.markdown(f"<h4 style='color:{color}'>{quality}</h4>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction error: {e}")
