import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained pipeline model
model = joblib.load("best_xgb_model.pkl")

# Load dataset to get default values for all required columns
try:
    data = pd.read_csv("Ameshousing.csv.csv")
except FileNotFoundError:
    st.error("❌ Dataset 'Ameshousing.csv' not found in the same directory as this app.")
    st.stop()

# Make a template input row with default values
input_df = data.iloc[0:1].copy()

# Streamlit UI
st.set_page_config(page_title="House Price Predictor", layout="centered")
st.title("🏠 Ames House Price Prediction App")
st.markdown("Enter property details below to estimate its sale price.")

# Accept user inputs to overwrite some features
input_df["Gr Liv Area"] = st.slider("Above Ground Living Area (sq ft)", 500, 5000, 1500)
input_df["Garage Cars"] = st.slider("Number of Garage Cars", 0, 4, 2)
input_df["Year Built"] = st.slider("Year Built", 1900, 2022, 2000)
input_df["Total Bsmt SF"] = st.slider("Total Basement SF", 0, 3000, 800)
input_df["Overall Qual"] = st.slider("Overall Quality (1–10)", 1, 10, 5)
input_df["Full Bath"] = st.slider("Full Bathrooms", 0, 4, 2)
input_df["Lot Area"] = st.slider("Lot Area (sq ft)", 2000, 200000, 8000)

# Fill any missing values with median (just in case)
input_df.fillna(data.median(numeric_only=True), inplace=True)

# Predict
if st.button("Predict House Price 💰"):
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"🏡 Estimated Sale Price: **${prediction:,.0f}**")
    except Exception as e:
        st.error(f"❌ Error making prediction: {e}")
