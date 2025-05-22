import streamlit as st
import pandas as pd
import pickle
import time

# Load model and preprocessing tools
with open("xgb_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)
with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)
with open("scalers.pkl", "rb") as f:
    scalers = pickle.load(f)

st.set_page_config(page_title="🏠 House Price Predictor")


# --- Simple Styling for clean warnings and animations ---
st.markdown("""
    <style>
    [data-testid="stHeader"], [data-testid="stToolbar"], #MainMenu, footer {
        visibility: hidden;
    }
    .warning-text {
        color: #d9534f;
        font-weight: 500;
        margin-top: -8px;
        margin-bottom: 10px;
    }
    .step-title {
        font-size: 28px;
        font-weight: bold;
        color: #1976D2;
        margin-bottom: 20px;
    }
    .shake {
        animation: shake 0.3s;
        animation-iteration-count: 1;
    }
    @keyframes shake {
        0% { transform: translateX(0); }
        25% { transform: translateX(-6px); }
        50% { transform: translateX(6px); }
        75% { transform: translateX(-4px); }
        100% { transform: translateX(0); }
    }
    </style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <a href="https://pakproperties20250521145834-fze9fuebhga8f0h7.canadacentral-01.azurewebsites.net/" target="_blank" >Back To Home Page</a>
    """,
    unsafe_allow_html=True,
)

# --- Title ---
st.markdown('<div class="step-title">🏠 Pakistan House Price Predictor</div>', unsafe_allow_html=True)

# --- Marla to Square Feet Convertor ---
with st.expander("📏 Convertor: Marla ➜ Square Feet"):
    st.markdown("👷 Enter size in **Marla** to convert to **Square Feet**.")
    marla_input = st.number_input("Enter size in Marla", min_value=0.0, step=0.5, key="marla_input")
    if st.button("🔁 Convert to Sq Ft"):
        sq_ft = marla_input * 272.25
        st.success(f"✅ {marla_input} Marla = **{sq_ft:,.2f} sq ft**")



# --- Inputs: Line-by-line ---
type_options = ["Select Property Type"] + list(encoders['type'].classes_)
type_input = st.selectbox("🏘️ Property Type", type_options)
if type_input == "Select Property Type":
    st.markdown('<div class="warning-text">⚠ Required</div>', unsafe_allow_html=True)

city_options = ["Select City"] + list(encoders['city'].classes_)
city_input = st.selectbox("🌆 City", city_options)
if city_input == "Select City":
    st.markdown('<div class="warning-text">⚠ Required</div>', unsafe_allow_html=True)

location_options = ["Select Location"] + list(encoders['location'].classes_)
location_input = st.selectbox("📍 Location", location_options)
if location_input == "Select Location":
    st.markdown('<div class="warning-text">⚠ Required</div>', unsafe_allow_html=True)

purpose_options = ["Select Purpose"] + list(encoders['purpose'].classes_)
purpose_input = st.selectbox("🎯 Purpose", purpose_options)
if purpose_input == "Select Purpose":
    st.markdown('<div class="warning-text">⚠ Required</div>', unsafe_allow_html=True)

beds_input = st.number_input("🛏️ Bedrooms", min_value=1, step=1)
if beds_input <= 0:
    st.markdown('<div class="warning-text">⚠ Must be >= 1</div>', unsafe_allow_html=True)

baths_input = st.number_input("🛁 Baths", min_value=1, step=1)
if baths_input <= 0:
    st.markdown('<div class="warning-text">⚠ Must be >= 1</div>', unsafe_allow_html=True)

area_input = st.number_input("📐 Area (sq ft)", min_value=1.0)
if area_input <= 0:
    st.markdown('<div class="warning-text">⚠ Must be >= 1</div>', unsafe_allow_html=True)


# --- Validation ---
form_valid = (
    type_input != "Select Property Type"
    and city_input != "Select City"
    and location_input != "Select Location"
    and purpose_input != "Select Purpose"
    and beds_input > 0
    and baths_input > 0
    and area_input > 1.0
)

# --- Predict Button ---
if st.button("🔮 Predict Price"):
    if not form_valid:
        st.warning("Please correct the highlighted fields.")
        st.markdown("""
            <script>
            const btn = window.parent.document.querySelector('button[kind="primary"]');
            btn.classList.add('shake');
            btn.scrollIntoView({behavior: 'smooth'});
            setTimeout(() => btn.classList.remove('shake'), 300);
            </script>
        """, unsafe_allow_html=True)
    else:
        with st.spinner("Predicting house price..."):
            time.sleep(1.2)

        # Encode + scale
        type_encoded = encoders['type'].transform([type_input])[0]
        city_encoded = encoders['city'].transform([city_input])[0]
        location_encoded = encoders['location'].transform([location_input])[0]
        purpose_encoded = encoders['purpose'].transform([purpose_input])[0]

        area_scaled = scalers['area'].transform([[area_input]])[0][0]
        baths_scaled = scalers['baths'].transform([[baths_input]])[0][0]
        beds_scaled = scalers['beds'].transform([[beds_input]])[0][0]

        input_df = pd.DataFrame({
            "type": [type_encoded],
            "location": [location_encoded],
            "city": [city_encoded],
            "purpose": [purpose_encoded],
            "baths": [baths_scaled],
            "beds": [beds_scaled],
            "area": [area_scaled]
        })

        prediction = xgb_model.predict(input_df)[0]

        st.success(f"🎉 Predicted House Price: **PKR {prediction:,.2f}**")
        st.balloons()
