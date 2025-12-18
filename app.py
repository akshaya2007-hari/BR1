import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# -----------------------------
# App Title
# -----------------------------
st.title("🏠 Real Estate Price Prediction App")
st.write("Polynomial Regression using Distance to MRT Station")

# -----------------------------
# Upload Dataset
# -----------------------------
uploaded_file = st.file_uploader("Real estate.csv", type=["csv"])

if uploaded_file is not None:

    # Load dataset
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # -----------------------------
    # Feature & Target
    # -----------------------------
    X = df[['X3 distance to the nearest MRT station']]
    y = df['Y house price of unit area']

    # -----------------------------
    # Polynomial Regression
    # -----------------------------
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)

    # -----------------------------
    # USER INPUT
    # -----------------------------
    st.subheader("🔢 Enter Input Value")

    distance = st.number_input(
        "Distance to nearest MRT station (in meters)",
        min_value=0.0,
        max_value=float(X.max()),
        step=10.0
    )

    # Convert input to polynomial form
    input_poly = poly.transform([[distance]])

    # Prediction
    predicted_price = model.predict(input_poly)

    st.success(
        f"💰 Predicted House Price per Unit Area: **{predicted_price[0]:.2f}**"
    )

    # -----------------------------
    # Plot Curve + Input Point
    # -----------------------------
    st.subheader("📈 Polynomial Regression Curve")

    X_sorted = X.sort_values(by='X3 distance to the nearest MRT station')
    X_sorted_poly = poly.transform(X_sorted)
    y_pred = model.predict(X_sorted_poly)

    fig, ax = plt.subplots()
    ax.scatter(X, y, label="Actual Data")
    ax.plot(X_sorted, y_pred, color="red", label="Polynomial Curve")
    ax.scatter(distance, predicted_price, color="green", s=100, label="Your Input")

    ax.set_xlabel("Distance to MRT Station")
    ax.set_ylabel("House Price per Unit Area")
    ax.set_title("Polynomial Regression (Degree = 2)")
    ax.legend()

    st.pyplot(fig)

else:
    st.warning("⬆️ Please upload the Real estate.csv file")
