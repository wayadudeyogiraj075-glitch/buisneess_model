import streamlit as st
import numpy as np
import pickle
import joblib

# Load model safely (pickle or joblib)
@st.cache_resource
def load_model():
    try:
        return joblib.load("model.pkl")
    except:
        with open("model.pkl", "rb") as f:
            return pickle.load(f)

model = load_model()

# UI
st.title("ML Prediction App")
st.write("Enter input features separated by commas")

# Input as text (flexible)
user_input = st.text_input("Input Features (e.g., 1,2,3,4)")

if st.button("Predict"):
    try:
        # Convert input string → list of floats
        features = [float(x.strip()) for x in user_input.split(",")]
        features_array = np.array(features).reshape(1, -1)

        prediction = model.predict(features_array)

        st.success(f"Prediction: {prediction[0]}")

    except Exception as e:
        st.error(f"Error: {e}")
