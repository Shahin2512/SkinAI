# app.py
import streamlit as st
import os
from predict import predict_image
from llama_recommender import get_suggestion

st.set_page_config(page_title="AI Skin Disease Analyzer", layout="centered")
st.title("🧬 AI-Based Skin Disease Detector")

uploaded_file = st.file_uploader("Upload a skin image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image("temp.jpg", caption="Uploaded Image", width=300)

    if st.button("🔍 Predict"):
        result = predict_image("temp.jpg")
        st.success(f"✅ Predicted Disease: {result}")

        st.subheader("💡 Recommendations")
        suggestions = get_suggestion(result)
        st.info(suggestions)
