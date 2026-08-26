# TASK10 — Deploy Your Model as a Live Web App

## What this is

A simple Streamlit web app that predicts Titanic passenger survival using the
best-performing pipeline from Task 7/9 (Logistic Regression inside a
scikit-learn `Pipeline` with `ColumnTransformer` + 2 engineered features:
`FamilySize`, `IsAlone`). Accuracy on held-out test data: **80.4%**
(F1-score: 0.729).

**Live app:** https://task10-dmsnjz2s25mb53dbgmyjza.streamlit.app/

## Files

- `app.py` — the Streamlit app: input fields for passenger details, a
  "Predict Survival" button, and a displayed prediction with confidence.
- `titanic_survival_pipeline.joblib` — the saved, trained pipeline
  (preprocessing + model in one object).
- `requirements.txt` — dependencies needed for Streamlit Cloud to install
  and run the app.
- `titanic.csv` — the dataset used to train the model (not required by the
  app itself, kept for reference/reproducibility).

## How to run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## How to deploy for free on Streamlit Community Cloud

1. Make sure this `TASK10` folder (with `app.py`, `titanic_survival_pipeline.joblib`,
   and `requirements.txt`) is pushed to your public GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with
   your GitHub account.
3. Click **"New app"**.
4. Select this repository: `Subhantrq27/neurofive-ml-track`.
5. Set the **branch** to `main`.
6. Set the **main file path** to `TASK10/app.py`.
7. Click **"Deploy"**. The first deploy takes a couple of minutes while it
   installs dependencies from `requirements.txt`.
8. Once live, copy the app's URL (something like
   `https://your-app-name.streamlit.app`) and paste it into the "Live app"
   line at the top of this README, then commit and push that update.

## What the app does

The user enters passenger details (class, sex, age, fare, port of
embarkation, siblings/spouses aboard, parents/children aboard). The app
computes `FamilySize` and `IsAlone` automatically, feeds everything through
the saved pipeline, and displays the predicted outcome (Survived / Did not
survive) along with the model's confidence for that prediction.
