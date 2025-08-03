# heart_app.py

import streamlit as st
import pandas as pd
import joblib

# Set page config
st.set_page_config(page_title="Heart Disease Predictor", page_icon="🫀", layout="centered")

# Inject 3D neumorphic-style CSS
st.markdown("""
<style>
    html, body, [class*="css"] {
        background-color: #1a1a1a;
        color: #f5f5f5;
        font-family: 'Segoe UI', sans-serif;
    }

    .stSlider > div {
        background-color: #262626;
        padding: 1rem;
        border-radius: 20px;
        box-shadow: 4px 4px 10px #0f0f0f, -4px -4px 10px #2b2b2b;
    }

    .stButton>button {
        background: linear-gradient(145deg, #1f1f1f, #252525);
        border: none;
        border-radius: 12px;
        color: white;
        font-weight: bold;
        box-shadow: 4px 4px 10px #121212, -4px -4px 10px #2a2a2a;
        transition: all 0.3s ease-in-out;
    }

    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 6px 6px 12px #101010, -6px -6px 12px #2e2e2e;
    }

    .stTitle, .stHeader {
        text-align: center;
        color: #ff4b4b;
    }
</style>
""", unsafe_allow_html=True)

# Load model and preprocessing files
model = joblib.load("knn.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# App title
st.title("🫀 Heart Disease Prediction App")

# Input fields
st.subheader("Enter Patient Data:")
age = st.slider("Age", 20, 90, 55)
sex = st.selectbox("Sex (1 = Male, 0 = Female)", [1, 0])
cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
trestbps = st.slider("Resting Blood Pressure", 80, 200, 120)
chol = st.slider("Cholesterol", 100, 400, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)", [1, 0])
restecg = st.selectbox("Rest ECG (0–2)", [0, 1, 2])
thalach = st.slider("Max Heart Rate Achieved", 60, 250, 150)
exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [1, 0])
oldpeak = st.slider("Oldpeak (ST depression)", 0.0, 6.0, 1.0, step=0.1)
slope = st.selectbox("Slope of ST Segment (0–2)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels Colored (0–4)", [0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia (0 = Normal, 1 = Fixed Defect, 2 = Reversible Defect)", [0, 1, 2])

# Predict button
if st.button("🚀 Predict"):
    input_data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }

    # DataFrame + One-hot encoding
    input_df = pd.DataFrame([input_data])
    input_df = pd.get_dummies(input_df)

    # Add missing columns
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Match column order
    input_df = input_df[feature_columns]

    # Scale selected features
    num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    input_df[num_cols] = scaler.transform(input_df[num_cols])

    # Make prediction
    prediction = model.predict(input_df)[0]
    result = "🛑 Positive for Heart Disease---Your life at risk" if prediction == 1 else "✅ Negative for Heart Disease---You are out of danger"

    st.markdown(f"### 💡 Prediction: {result}")
