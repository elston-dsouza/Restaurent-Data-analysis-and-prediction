import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

st.set_page_config(layout="wide")

scaler = joblib.load("Scaler.pkl")

st.title("Restaurent Rating Prediction")


st.caption("Predicts restaurent reviews")

st.divider()

averagecost = st.number_input("Please enter the estimated average cost for two", min_value=50,max_value=99999,value=1000,step=200)

tablebooking = st.selectbox("restaurent has table booking?",["Yes","No"])

Onlinedelivery = st.selectbox("restaurent has Online delivery?",["Yes","No"])

Pricerange = st.selectbox("What is the price range(1 is the Cheapest, 4 is the most expensive)",[1,2,3,4])

predictbttn = st.button("predict the review!")

st.divider()

model = joblib.load("mlmodel.pkl")

bookingstatus = 1 if tablebooking == "yes" else 0

deliverystatus = 1 if Onlinedelivery == "yes" else 0

values = [[averagecost,bookingstatus,deliverystatus,Pricerange]]
my_X_values = np.array(values)

X = scaler.transform(my_X_values)

if predictbttn:
    st.snow()
    
    prediction = model.predict(X)
    
    if prediction < 2.5:
        st.write("poor")
    elif prediction < 3.5:
        st.write("Average")
    elif prediction < 4.0:
        st.write("Good")
    elif prediction < 4.5:
        st.write("Very Good")
    else:
        st.write("Excellent")