import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")

# Resolve the model path relative to this script's own location, so it
# works no matter what the current working directory is when Streamlit
# Cloud (or anyone else) launches the app.
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "titanic_survival_pipeline.joblib")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

st.title("🚢 Titanic Survival Predictor")
st.write(
    "This app uses a Logistic Regression model (trained inside a scikit-learn "
    "`Pipeline` with `ColumnTransformer`) to predict whether a passenger would "
    "have survived the Titanic disaster, based on their details. "
    "Model accuracy on held-out test data: **80.4%** (F1-score: 0.73)."
)

st.divider()
st.subheader("Enter passenger details")

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox(
        "Passenger Class", options=[1, 2, 3], index=2,
        format_func=lambda x: f"{x} ({'1st' if x==1 else '2nd' if x==2 else '3rd'} class)"
    )
    sex = st.selectbox("Sex", options=["male", "female"])
    age = st.slider("Age", min_value=0, max_value=80, value=30)
    fare = st.number_input("Fare paid (£)", min_value=0.0, max_value=600.0, value=32.0, step=1.0)

with col2:
    embarked = st.selectbox(
        "Port of Embarkation", options=["S", "C", "Q"],
        format_func=lambda x: {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}[x]
    )
    sibsp = st.number_input("Siblings/Spouses aboard", min_value=0, max_value=10, value=0)
    parch = st.number_input("Parents/Children aboard", min_value=0, max_value=10, value=0)

family_size = sibsp + parch + 1
is_alone = 1 if family_size == 1 else 0

st.caption(f"Computed: Family size = {family_size}, Traveling alone = {'Yes' if is_alone else 'No'}")

st.divider()

if st.button("Predict Survival", type="primary", use_container_width=True):
    input_df = pd.DataFrame([{
        "Age": age,
        "Fare": fare,
        "SibSp": sibsp,
        "Parch": parch,
        "Pclass": pclass,
        "FamilySize": family_size,
        "IsAlone": is_alone,
        "Sex": sex,
        "Embarked": embarked,
    }])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    if prediction == 1:
        st.success(f"✅ **Predicted: Survived** (confidence: {probability[1]*100:.1f}%)")
    else:
        st.error(f"❌ **Predicted: Did not survive** (confidence: {probability[0]*100:.1f}%)")

    st.progress(float(probability[1]))
    st.caption(f"Survival probability: {probability[1]*100:.1f}%")

st.divider()
st.caption(
    "Built as part of the Neurofive Solutions Machine Learning Fundamentals track. "
    "[View the full project on GitHub](https://github.com/Subhantrq27/neurofive-ml-track)"
)
