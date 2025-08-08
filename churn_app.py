# ─────────────────────────────────────────────
# 🧠 IMPORTS & MODEL LOADING
# ─────────────────────────────────────────────
import streamlit as st
import pandas as pd
import pickle
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

# ─────────────────────────────────────────────
# 📋 SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.title("📋 About this App")
st.sidebar.markdown("""
This AI dashboard predicts customer churn using an XGBoost model.

- Upload a preprocessed customer CSV
- Get churn probabilities and predictions
- Visualize churn insights

🔗 [GitHub Repo](https://github.com/your-repo)
""")

# ─────────────────────────────────────────────
# 📊 MAIN CONTENT
# ─────────────────────────────────────────────
st.title("📊 AI-Powered Customer Churn Dashboard")

uploaded_file = st.file_uploader("Upload a preprocessed CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("👁️ Customer Data Preview")
    st.dataframe(df.head())

    # ─────────────────────────────────────────
    # 🔍 CHURN PREDICTIONS
    # ─────────────────────────────────────────
    preds = model.predict_proba(df)[:, 1]
    df["Churn Probability"] = preds
    df["Churn Prediction"] = (df["Churn Probability"] > 0.5).astype(int)

    st.subheader("🔮 Prediction Results")
    styled_df = df[["Churn Probability", "Churn Prediction"]].style \
        .background_gradient(subset=["Churn Probability"], cmap='Reds') \
        .applymap(lambda val: 'color: red' if val == 1 else 'color: green', subset=["Churn Prediction"])
    st.dataframe(styled_df)

    # ─────────────────────────────────────────
    # 📉 CHURN DISTRIBUTION INSIGHTS
    # ─────────────────────────────────────────
    if 'Churn' in df.columns:
        st.subheader("📉 Churn Distribution by Contract Type")
        fig1, ax1 = plt.subplots()
        sns.barplot(x='Contract', hue='Churn', data=df, ax=ax1)
        st.pyplot(fig1)

        st.subheader("💳 Churn by Payment Method")
        fig2, ax2 = plt.subplots()
        sns.barplot(x='PaymentMethod', hue='Churn', data=df, ax=ax2)
        plt.xticks(rotation=45)
        st.pyplot(fig2)

    # ─────────────────────────────────────────
    # 📌 FEATURE IMPORTANCE
    # ─────────────────────────────────────────
   t.subheader("📌 Top Churn Drivers")

feature_names = model.get_booster().feature_names
importances = model.feature_importances_

features_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

fig3, ax3 = plt.subplots()
sns.barplot(x='Importance', y='Feature', data=features_df.head(10), ax=ax3)
st.pyplot(fig3)
    # ─────────────────────────────────────────
    # 📥 DOWNLOAD PREDICTIONS
    # ─────────────────────────────────────────
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Predictions as CSV",
        data=csv,
        file_name="churn_predictions.csv",
        mime="text/csv"
    )
else:
    st.info("Upload a CSV file to get started.")

