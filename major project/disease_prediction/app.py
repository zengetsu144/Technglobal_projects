"""
Disease Prediction System - Streamlit app
Run:  streamlit run app.py   (after running train.py once)
"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from common import clean_diabetes

MODEL_DIR = Path(__file__).parent / "models"
st.set_page_config(page_title="Disease Prediction System", page_icon="🩺")


@st.cache_resource
def load_bundle(name):
    path = MODEL_DIR / f"{name}.joblib"
    return joblib.load(path) if path.exists() else None


def show_result(bundle, row, label):
    row = row[bundle["features"]]
    p = float(bundle["pipeline"].predict_proba(row)[0, 1])
    st.subheader("Result")
    if p >= 0.5:
        st.error(f"Model predicts a **higher likelihood** of {label}.")
    else:
        st.success(f"Model predicts a **lower likelihood** of {label}.")
    st.metric("Estimated probability", f"{p:.1%}")
    st.progress(min(max(p, 0.0), 1.0))
    st.caption("This is a screening aid built for learning, not a diagnosis. "
               "Please consult a doctor for any health concern.")


def show_model_info(bundle):
    with st.expander("About this model"):
        m = bundle["metrics"]
        st.write(f"**Model:** {bundle['model_name']}")
        st.write(f"**Features used:** {', '.join(bundle['features_used'])}")
        c = st.columns(4)
        c[0].metric("Accuracy", f"{m['test_accuracy']:.2f}")
        c[1].metric("Recall", f"{m['test_recall']:.2f}")
        c[2].metric("F1", f"{m['test_f1']:.2f}")
        c[3].metric("ROC-AUC", f"{m['test_auc']:.2f}")
        st.caption("Metrics are from a held-out test set.")


st.title("🩺 Disease Prediction System")
disease = st.sidebar.radio("Choose condition", ["Diabetes", "Heart Disease"])
st.sidebar.info("Educational project. Not medical advice.")

if disease == "Diabetes":
    bundle = load_bundle("diabetes")
    if bundle is None:
        st.error("Model not found. Run `python train.py` first.")
        st.stop()
    st.header("Diabetes risk (PIMA dataset)")
    with st.form("diabetes"):
        c1, c2 = st.columns(2)
        preg = c1.number_input("Pregnancies", 0, 20, 1)
        glu = c2.number_input("Glucose (mg/dL)", 0, 300, 120)
        bp = c1.number_input("Blood pressure (mm Hg)", 0, 200, 70)
        skin = c2.number_input("Skin thickness (mm)", 0, 100, 20)
        ins = c1.number_input("Insulin (mu U/ml)", 0, 900, 80)
        bmi = c2.number_input("BMI", 0.0, 70.0, 28.0, step=0.1)
        dpf = c1.number_input("Diabetes pedigree function", 0.0, 2.5, 0.5, step=0.01)
        age = c2.number_input("Age", 1, 100, 33)
        go = st.form_submit_button("Predict", type="primary")
    st.caption("Enter 0 for glucose, blood pressure, skin thickness, insulin or BMI if unknown; "
               "the model treats 0 as missing.")
    if go:
        row = clean_diabetes(pd.DataFrame([{
            "Pregnancies": preg, "Glucose": glu, "BloodPressure": bp, "SkinThickness": skin,
            "Insulin": ins, "BMI": bmi, "DiabetesPedigreeFunction": dpf, "Age": age}]))
        show_result(bundle, row, "diabetes")
    show_model_info(bundle)

else:
    bundle = load_bundle("heart")
    if bundle is None:
        st.error("Heart model not found. Put `heart.csv` in the `data/` folder and run `python train.py`.")
        st.stop()
    st.header("Heart disease risk (UCI / Cleveland data)")
    st.caption("Category codes follow the dataset (Kaggle heart.csv).")
    with st.form("heart"):
        c1, c2 = st.columns(2)
        age = c1.number_input("Age", 1, 120, 54)
        sex = c2.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male", index=1)
        cp = c1.selectbox("Chest pain type (cp)", [0, 1, 2, 3])
        trestbps = c2.number_input("Resting blood pressure (trestbps)", 80, 220, 130)
        chol = c1.number_input("Cholesterol mg/dl (chol)", 100, 600, 240)
        fbs = c2.selectbox("Fasting blood sugar > 120 mg/dl (fbs)", [0, 1],
                           format_func=lambda x: "No" if x == 0 else "Yes")
        restecg = c1.selectbox("Resting ECG (restecg)", [0, 1, 2])
        thalach = c2.number_input("Max heart rate achieved (thalach)", 60, 220, 150)
        exang = c1.selectbox("Exercise-induced angina (exang)", [0, 1],
                             format_func=lambda x: "No" if x == 0 else "Yes")
        oldpeak = c2.number_input("ST depression (oldpeak)", 0.0, 7.0, 1.0, step=0.1)
        slope = c1.selectbox("ST slope (slope)", [0, 1, 2], index=1)
        ca = c2.selectbox("Major vessels colored (ca)", [0, 1, 2, 3, 4])
        thal = c1.selectbox("Thalassemia (thal)", [0, 1, 2, 3], index=2)
        go = st.form_submit_button("Predict", type="primary")
    if go:
        row = pd.DataFrame([{
            "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol, "fbs": fbs,
            "restecg": restecg, "thalach": thalach, "exang": exang, "oldpeak": oldpeak,
            "slope": slope, "ca": ca, "thal": thal}])
        show_result(bundle, row, "heart disease")
    show_model_info(bundle)