import json

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Used Car Price Predictor", page_icon="🚗")

@st.cache_resource
def load():
    return joblib.load("models/model.joblib"), json.load(open("models/meta.json"))

model, meta = load()
m = meta["metrics"]

st.title("🚗 Used Car Price Predictor (UK)")
st.write("Enter a car's details to get an estimated resale price in £.")

with st.sidebar:
    st.header("About")
    st.write("XGBoost model trained on UK used-car listings.")
    st.metric("Test MAE", f"£{m['test_mae']:,.0f}")
    st.metric("Test R²", m["test_r2"])

c1, c2 = st.columns(2)
brand = c1.selectbox("Brand", meta["cat"]["brand"])
car_model = c2.selectbox("Model", meta["brand_models"][brand])
year = c1.slider("Year", int(meta["num"]["year"]["min"]), int(meta["num"]["year"]["max"]), 2017)
mileage = c2.number_input("Mileage", 0, 300000, 30000, step=1000)
transmission = c1.selectbox("Transmission", meta["cat"]["transmission"])
fuel = c2.selectbox("Fuel type", meta["cat"]["fuelType"])
engine = c1.number_input("Engine size (L)", 0.6, 6.5, 1.6, step=0.1)
tax = c2.number_input("Road tax (£)", 0, 600, 145, step=5)
mpg = c1.number_input("MPG", 5.0, 200.0, 50.0, step=1.0)

if st.button("Predict price", type="primary"):
    row = pd.DataFrame([{
        "model": car_model, "year": year, "transmission": transmission,
        "mileage": mileage, "fuelType": fuel, "tax": tax, "mpg": mpg,
        "engineSize": engine, "brand": brand,
    }])
    price = model.predict(row)[0]
    st.success(f"Estimated price: **£{price:,.0f}**")
    st.caption(f"Typical error is about £{m['test_mae']:,.0f}. Real prices depend on condition and market.")
