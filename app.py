import streamlit as st
import pandas as pd
import joblib

# Title
st.header("Educare Workshop")

# Input bar 1
height = st.number_input("Enter Height")

# Input bar 2
weight = st.number_input("Enter Weight")

# Dropdown input
eyes = st.selectbox("Select Eye Colour", ("Blue", "Brown"))

# If button is pressed
if st.button("Submit"):

    # Unpickle classifier
    clf = joblib.load("clf.pkl")

    # Store inputs into dataframe
    X = pd.DataFrame([[height, weight, eyes]],
                     columns=["Height", "Weight", "Eye"])
    X = X.replace(["Brown", "Blue"], [1, 0])

    # Get prediction
    # print("asdfasdfasdf")
    prediction = clf.predict(X)[0]

    # Output prediction
    st.text(f"This instance is a {prediction}")
