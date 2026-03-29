import streamlit as st
import pandas as pd
from src.predict import predict_customer

st.title("Bank Marketing Decision System")

st.write("Thông tin khách hàng:")

age = st.slider("Age", 18, 90, 30)
campaign = st.number_input("Số lần liên hệ", 1, 10, 2)

if st.button("Predict"):

    df = pd.DataFrame([{
        "age": age,
        "campaign": campaign,
        "job": "technician",
        "marital": "married",
        "education": "university.degree",
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "month": "may",
        "day_of_week": "mon",
        "pdays": 999,
        "previous": 0,
        "poutcome": "nonexistent",
        "emp.var.rate": -1.8,
        "cons.price.idx": 92.893,
        "cons.conf.idx": -46.2,
        "euribor3m": 1.313,
        "nr.employed": 5099.1
    }])

    result = predict_customer(df)

    st.subheader("Kết quả")

    st.write(f"Probability: {result['probability']:.2%}")
    st.write(f"Expected Profit: ${result['expected_profit']:.2f}")
    st.write(f"Action: {result['action']}")