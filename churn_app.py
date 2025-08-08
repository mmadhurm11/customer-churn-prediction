import streamlit as st
import pandas as pd
import pickle
import xgboost as xgb

# Page config
st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

st.title("AI-Powered Customer Churn Prediction Dashboard")

# Load model
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# Upload input file
uploaded_file = st.file_uploader("Upload a preprocessed customer CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader(" Customer Data Preview")
    st.dataframe(df.head())

    # Predict churn probability
    st.subheader(" Churn Predictions")
    preds = model.predict_proba(df)[:, 1]  # Get churn probabilities
    df["Churn Probability"] = preds
    df["Churn Prediction"] = (df["Churn Probability"] > 0.5).astype(int)

    st.dataframe(df[["Churn Probability", "Churn Prediction"]])

    # Download button
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=" Download Predictions as CSV",
        data=csv,
        file_name="churn_predictions.csv",
        mime="text/csv"
    )
else:
    st.info("Please upload a CSV file that matches your model's input format.")
