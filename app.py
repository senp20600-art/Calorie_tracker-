import streamlit as st # pyright: ignore[reportMissingImports]
import joblib # pyright: ignore[reportMissingImports]
import numpy as np# pyright: ignore[reportMissingImports]
calorie_model=joblib.load('calorie_predictor_model.pkl')
st.title("Calorie Burned Predictor")
st.write("Enter your activity details to predict your calorie burn:")
steps = st.number_input("Steps", min_value=0, step=1, format="%d")
distance = st.number_input("Distance (km)", min_value=0.0, step=0.1, format="%.2f")
logged_activities = st.number_input("Logged Activities", min_value=0, step=1, format="%d")
if st.button("Predict Calorie Burn"):
    input_data = np.array([[steps, distance, logged_activities]])
    predicted_calories = calorie_model.predict(input_data)[0]
    st.success(f"Estimated Calorie Burn: {predicted_calories:.2f} calories")
