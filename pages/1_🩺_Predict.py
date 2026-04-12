import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Predict Leukemia", page_icon="🩺")

st.title("🔬 Leukemia Prediction")
st.markdown("Upload gene expression data")

@st.cache_resource
def load_model():
    with open('models/leukemia_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

uploaded = st.file_uploader("Choose CSV file", type=["csv", "txt"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.write(f"Loaded {df.shape[1]} genes")
    
    if st.button("Predict"):
        data = df.values.flatten()[:5000]
        if len(data) < 5000:
            data = np.pad(data, (0, 5000 - len(data)))
        scaled = scaler.transform(data.reshape(1, -1))
        pred = model.predict(scaled)[0]
        st.success(f"**Prediction: {pred}**")