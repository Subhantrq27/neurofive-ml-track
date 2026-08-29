# Capstone — Forecasting Smog Before It Happens (Pakistan Air Quality Prediction)

**Live app:** https://neurofive-ml-track-emehttm4izvhuzmnfemywv.streamlit.app/

## Problem Statement

Every winter, cities across Punjab (Lahore, Faisalabad, Multan, Sialkot in
particular) are blanketed in hazardous smog. Air quality monitoring tells
people how bad the air is **right now** — but by the time PM2.5 sensors
report "Hazardous," it's already too late to plan around it. Weather
forecasts, on the other hand, are available a day or more in advance.

**Can tomorrow's air quality be predicted using only weather forecast
data** — temperature, humidity, wind, pressure, precipitation, time of
year — **without** real-time pollution sensors? If so, schools, hospitals,
outdoor workers, and families with asthma/allergies could plan a day ahead,
the same way people check a rain forecast before making outdoor plans.

## Dataset

Hourly air quality + weather readings across **10 Pakistani cities**
(Lahore, Karachi, Islamabad, Rawalpindi, Faisalabad, Multan, Peshawar,
Quetta, Sialkot, Rahim Yar Khan), **Nov 2025 – Feb 2026** (21,840 rows) —
covering the peak smog season. Source: Kaggle (Pakistan air quality +
weather dataset).

## Approach

1. **Cleaning** — the dataset arrived clean (no missing values, no
   duplicates).
2. **Critical design decision — avoiding leakage:** `aqi_category` in the
   raw data is directly derived from `pm2_5` via standard EPA breakpoints
   (verified in the notebook). Using `pm2_5`/`pm10`/other pollutant
   readings as model inputs to predict the AQI category would be leakage —
   a trivial, useless model that only works because the sensor already
   measured the pollution. Instead, this project predicts **PM2.5
   concentration using only weather + time features** — the same inputs a
   real forecast-based product would have available *before* the pollution
   happens.
3. **EDA** — PM2.5 varies sharply by city (industrial Punjab cities like
   Faisalabad/Multan/Lahore/Sialkot are far worse than Quetta/Karachi), by
   hour of day, and correlates with wind speed and humidity (calm, humid
   air traps pollution; wind disperses it).
4. **Feature engineering** — cyclical sine/cosine encoding for hour and
   month (so hour 23 and hour 0 are treated as adjacent), an `is_rush_hour`
   flag (7-9 AM, 5-7 PM), city one-hot encoded.
5. **Models trained and compared:** Linear Regression, Random Forest,
   XGBoost — all inside a single `sklearn.pipeline.Pipeline` with
   `ColumnTransformer` (`StandardScaler` for numerical features,
   `OneHotEncoder` for city).
6. **Best model selected** by R² on held-out test data, saved with
   `joblib`.
7. **Deployed** as a Streamlit app: enter a weather forecast, get a
   predicted PM2.5 value converted into a friendly AQI category with a
   health advisory.

## Results

| Model              | RMSE    | MAE     | R²     |
|---------------------|:-------:|:-------:|:------:|
| **XGBoost** (best)  | 24.96   | 16.69   | **0.858** |
| Random Forest        | 27.48   | 18.43   | 0.828  |
| Linear Regression     | 42.91   | 30.82   | 0.579  |

XGBoost explains **~86% of the variation in PM2.5** using only weather and
time features — no pollution sensor data. Tree-based models substantially
outperform Linear Regression, since the relationship between weather and
pollution is highly non-linear (e.g. wind's effect on dispersal isn't a
straight line, and city identity interacts with season in complex ways).

**Top predictors:** city identity (each city has a very different baseline
pollution level — Quetta and Faisalabad are worlds apart), hour of day
(cyclical encoding), wind speed, and whether it's rush hour.

## What does R² = 0.86 mean, in plain English?

Out of everything that determines how polluted the air will be, our model
can explain about 86% of it using just the weather forecast and the time of
day — a genuinely strong result for a real-world environmental prediction
problem. The remaining ~14% comes from things weather can't capture: how
much crop-burning or traffic happens on a given day, industrial activity,
and other day-to-day variation. In practice, this means the app's forecast
is a reliable guide for planning — "expect rough air tomorrow" — even
though it won't be exact down to the last decimal.

## Business / Real-World Value

Pakistan's winter smog crisis affects tens of millions of people, and public
health guidance is almost always reactive — issued only after air quality
has already turned hazardous. This project flips that: because it needs
**only weather forecast inputs**, it can generate a smog forecast a full
day ahead, using the weather data that's already collected and published
daily. A tool like this could plausibly:

- Let **schools** decide the night before whether to hold outdoor activities.
- Give **hospitals and clinics** a heads-up to prepare for a likely rise in
  respiratory complaints.
- Help **individuals with asthma or allergies** plan medication and outdoor
  exposure a day in advance, not react after symptoms start.
- Support **local government advisories** (e.g. odd-even traffic rules,
  school closures) with a forward-looking, data-driven trigger instead of a
  same-day reactive one.

The core insight — that a genuinely useful smog forecast doesn't need
expensive real-time pollution sensors, just the weather forecast that's
already available — is what makes this practical to actually deploy
cheaply in under-resourced settings.

## How to run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## How to deploy on Streamlit Community Cloud

1. Push this `TASK11` folder (with `app.py`, `air_quality_pipeline.joblib`,
   `requirements.txt`) to your public GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.
3. Click **"New app"**.
4. Repository: `Subhantrq27/neurofive-ml-track`, branch: `main`,
   main file path: `TASK11/app.py`.
5. Click **"Deploy"**.
6. Once live, copy the app URL and paste it into the "Live app" line at the
   top of this README, then commit and push that update.

## Files

- `air_quality_capstone.ipynb` — full notebook: problem statement, EDA,
  leakage check, feature engineering, multi-model comparison, best model
  selection, and results summary.
- `app.py` — the Streamlit forecasting app.
- `air_quality_pipeline.joblib` — the saved, trained pipeline
  (preprocessing + XGBoost model).
- `requirements.txt` — dependencies for Streamlit Cloud.
- `air_quality_pakistan.csv` — the dataset used.
