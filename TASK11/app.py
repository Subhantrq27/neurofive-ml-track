import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Pakistan Smog Forecaster", page_icon="🌫️", layout="centered")

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "air_quality_pipeline.joblib")

CITIES = [
    "Faisalabad", "Islamabad", "Karachi", "Lahore", "Multan",
    "Peshawar", "Quetta", "Rahim Yar Khan", "Rawalpindi", "Sialkot"
]

CATEGORY_COLORS = {
    "Good": "#4C9A5A",
    "Moderate": "#C9A227",
    "Unhealthy for Sensitive Groups": "#D97B29",
    "Unhealthy": "#C0392B",
    "Very Unhealthy": "#7B2D8E",
    "Hazardous": "#5C0A0A",
}


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def pm25_to_category(pm25):
    if pm25 <= 12.0:
        return "Good", "Air quality is satisfactory. Enjoy outdoor activities."
    elif pm25 <= 35.4:
        return "Moderate", "Acceptable air quality. Unusually sensitive people should consider reducing prolonged outdoor exertion."
    elif pm25 <= 55.4:
        return "Unhealthy for Sensitive Groups", "Sensitive groups (children, elderly, asthma/heart patients) should reduce prolonged outdoor exertion."
    elif pm25 <= 150.4:
        return "Unhealthy", "Everyone may experience health effects. Limit prolonged outdoor exertion, especially sensitive groups."
    elif pm25 <= 250.4:
        return "Very Unhealthy", "Health alert: everyone may experience more serious health effects. Avoid outdoor activity."
    else:
        return "Hazardous", "Health emergency. Everyone should avoid outdoor exertion; stay indoors with air filtration if possible."


model = load_model()

st.title("🌫️ Pakistan Smog Forecaster")
st.write(
    "Predicts tomorrow's PM2.5 air quality using **only weather forecast data** — "
    "no pollution sensors required. Trained on hourly data across 10 Pakistani "
    "cities through the Nov 2025 – Feb 2026 smog season. "
    "**Model: XGBoost, R² = 0.86 on held-out test data.**"
)

st.divider()
st.subheader("Enter tomorrow's forecast")

col1, col2 = st.columns(2)

with col1:
    city = st.selectbox("City", options=CITIES, index=CITIES.index("Lahore"))
    month = st.selectbox(
        "Month", options=list(range(1, 13)), index=11,
        format_func=lambda m: ["January", "February", "March", "April", "May", "June",
                                "July", "August", "September", "October", "November", "December"][m-1]
    )
    hour = st.slider("Hour of day", min_value=0, max_value=23, value=8)
    is_weekend = st.checkbox("Weekend (Sat/Sun)?")

with col2:
    temperature = st.slider("Forecast temperature (°C)", min_value=-5.0, max_value=45.0, value=15.0)
    humidity = st.slider("Forecast humidity (%)", min_value=0.0, max_value=100.0, value=70.0)
    wind_speed = st.slider("Forecast wind speed (m/s)", min_value=0.0, max_value=20.0, value=2.0)
    pressure = st.slider("Forecast pressure (hPa)", min_value=980.0, max_value=1040.0, value=1018.0)

precipitation = st.slider("Forecast precipitation (mm)", min_value=0.0, max_value=50.0, value=0.0)
wind_direction = st.slider("Forecast wind direction (°)", min_value=0.0, max_value=360.0, value=180.0)

is_rush_hour = 1 if hour in [7, 8, 9, 17, 18, 19] else 0
hour_sin = np.sin(2 * np.pi * hour / 24)
hour_cos = np.cos(2 * np.pi * hour / 24)
month_sin = np.sin(2 * np.pi * month / 12)
month_cos = np.cos(2 * np.pi * month / 12)

st.divider()

if st.button("Forecast Air Quality", type="primary", use_container_width=True):
    input_df = pd.DataFrame([{
        "temperature": temperature,
        "humidity": humidity,
        "precipitation": precipitation,
        "wind_speed": wind_speed,
        "wind_direction": wind_direction,
        "pressure": pressure,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "is_rush_hour": is_rush_hour,
        "is_weekend": int(is_weekend),
        "city": city,
    }])

    predicted_pm25 = float(model.predict(input_df)[0])
    predicted_pm25 = max(predicted_pm25, 0)
    category, advice = pm25_to_category(predicted_pm25)
    color = CATEGORY_COLORS[category]

    st.markdown(
        f"""
        <div style="padding: 20px; border-radius: 10px; background-color: {color}20; border-left: 6px solid {color};">
            <h3 style="margin:0; color:{color};">{category}</h3>
            <p style="margin:6px 0 0 0; font-size: 1.1em;">Predicted PM2.5: <b>{predicted_pm25:.1f} µg/m³</b></p>
            <p style="margin:8px 0 0 0;">{advice}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    max_scale = 300
    st.progress(min(predicted_pm25 / max_scale, 1.0))
    st.caption(f"On a 0–{max_scale}+ µg/m³ scale (Hazardous starts at 250.5)")

st.divider()
st.caption(
    "Capstone project for the Neurofive Solutions Machine Learning Fundamentals track. "
    "Predicts PM2.5 from weather-forecast-only inputs (no pollutant sensor data), "
    "so the same forecast pipeline could realistically run a day in advance. "
    "[View the full project on GitHub](https://github.com/Subhantrq27/neurofive-ml-track)"
)
