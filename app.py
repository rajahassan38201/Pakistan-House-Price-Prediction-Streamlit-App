import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the model, encoders, and scalers
with open("xgb_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

with open("scalers.pkl", "rb") as f:
    scalers = pickle.load(f)


# Streamlit App
st.set_page_config(
    page_title="House Price Prediction App",
    layout="wide"  # Use wide layout to utilize full screen
)

# Remove header and footer
st.markdown(
    """
    <style>
    .reportview-container {
        margin-top: -80px;
    }
   [data-testid="stHeader"], [data-testid="stToolbar"] {
        display: none;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <a href="https://localhost:7067" target="_blank" >Back To Home Page</a>
    """,
    unsafe_allow_html=True,
)
# Streamlit App
st.title("PAKISTAN | House Price Prediction")

# Input fields
type_options = list(encoders['type'].classes_)
type_input = st.selectbox("Property Type", type_options)

city_options = list(encoders['city'].classes_)
city_input = st.selectbox("City", city_options)

location_options = list(encoders['location'].classes_)
location_input = st.selectbox("Location", location_options)

purpose_options = list(encoders['purpose'].classes_)
purpose_input = st.selectbox("Purpose", purpose_options)

baths_input = st.number_input("Number of Baths", min_value=1, step=1)
beds_input = st.number_input("Number of Bedrooms", min_value=1, step=1)
area_input = st.number_input("Area (sq ft)", min_value=1.0)

# Prediction button
if st.button("Predict Price"):
    # Encode categorical features
    type_encoded = encoders['type'].transform([type_input])[0]
    city_encoded = encoders['city'].transform([city_input])[0]
    location_encoded = encoders['location'].transform([location_input])[0]
    purpose_encoded = encoders['purpose'].transform([purpose_input])[0]

    # Scale numerical features
    area_scaled = scalers['area'].transform([[area_input]])[0][0]
    baths_scaled = scalers['baths'].transform([[baths_input]])[0][0]
    beds_scaled = scalers['beds'].transform([[beds_input]])[0][0]

    # Create input DataFrame
    input_data = pd.DataFrame({
        "type": [type_encoded],
        "location": [location_encoded],
        "city": [city_encoded],
        "purpose": [purpose_encoded],
        "baths": [baths_scaled],
        "beds": [beds_scaled],
        "area": [area_scaled]
    })

    # Make prediction
    prediction = xgb_model.predict(input_data)[0]

    # Display prediction
    st.success(f"Predicted House Price: PKR {prediction:,.2f}")
