# Import required libraries
import streamlit as st
import pickle
import numpy as np
import pandas as pd
from PIL import Image
import os

# Set the page configuration
st.set_page_config(page_title="Timelytics", page_icon=":pencil:", layout="wide")

# Title and caption
st.title("Timelytics: Optimize your supply chain with advanced forecasting techniques.")

st.caption(
    "Timelytics is an ensemble model that utilizes three powerful machine learning algorithms - XGBoost, Random Forests, and Support Vector Machines (SVM) - to accurately forecast Order to Delivery (OTD) times."
)

st.caption(
    "With Timelytics, businesses can identify potential bottlenecks and delays in their supply chain and take proactive measures to reduce lead times and improve delivery."
)

# Load the model safely using os.path and error handling
model_path = os.path.join("voting_model.pkl")

try:
    with open(model_path, "rb") as file:
        voting_model = pickle.load(file)
except FileNotFoundError:
    st.error("Model file not found. Please make sure 'voting_model.pkl' exists in the root folder.")
    st.stop()

# Cache the predictor function
@st.cache_resource
def waitime_predictor(
    purchase_dow,
    purchase_month,
    year,
    product_size_cm3,
    product_weight_g,
    geolocation_state_customer,
    geolocation_state_seller,
    distance,
):
    prediction = voting_model.predict(
        np.array(
            [[
                purchase_dow,
                purchase_month,
                year,
                product_size_cm3,
                product_weight_g,
                geolocation_state_customer,
                geolocation_state_seller,
                distance,
            ]]
        )
    )
    return round(prediction[0])

# Sidebar input UI
with st.sidebar:
    image_path = os.path.join("assets", "supply_chain_optimisation.jpg")
    try:
        img = Image.open(image_path)
        st.image(img)
    except FileNotFoundError:
        st.warning("Image not found. Please ensure it exists at assets/supply_chain_optimisation.jpg")

    st.header("Input Parameters")
    purchase_dow = st.number_input("Purchased Day of the Week", min_value=0, max_value=6, step=1, value=3)
    purchase_month = st.number_input("Purchased Month", min_value=1, max_value=12, step=1, value=1)
    year = st.number_input("Purchased Year", value=2018)
    product_size_cm3 = st.number_input("Product Size in cm³", value=9328)
    product_weight_g = st.number_input("Product Weight in grams", value=1800)
    geolocation_state_customer = st.number_input("Geolocation State of the Customer", value=10)
    geolocation_state_seller = st.number_input("Geolocation State of the Seller", value=20)
    distance = st.number_input("Distance", value=475.35)
    submit = st.button(label="Predict Wait Time!")

# Prediction output
with st.container():
    st.header("Output: Wait Time in Days")
    if submit:
        with st.spinner("This may take a moment..."):
            prediction = waitime_predictor(
                purchase_dow,
                purchase_month,
                year,
                product_size_cm3,
                product_weight_g,
                geolocation_state_customer,
                geolocation_state_seller,
                distance,
            )
        st.success(f"Predicted Wait Time: {prediction} days")

    # Display sample dataset
    data = {
        "Purchased Day of the Week": [0, 3, 1],
        "Purchased Month": [6, 3, 1],
        "Purchased Year": [2018, 2017, 2018],
        "Product Size in cm³": [37206.0, 63714, 54816],
        "Product Weight in grams": [16250.0, 7249, 9600],
        "Geolocation State Customer": [25, 25, 25],
        "Geolocation State Seller": [20, 7, 20],
        "Distance": [247.94, 250.35, 4.915],
    }

    df = pd.DataFrame(data)
    st.header("Sample Dataset")
    st.write(df)
