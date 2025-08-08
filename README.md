
# Customer Churn Prediction & Retention Strategy 📉📈

This project uses machine learning (XGBoost) to predict customer churn based on telecom usage patterns. It also identifies key churn drivers to support business retention strategies.

## 🔍 Problem Statement

Customer churn is a major issue in telecom. The goal is to:
- Predict which customers are likely to leave
- Understand why they leave
- Propose actionable retention strategies

## 🧠 Techniques Used

- Data cleaning & preprocessing (pandas, LabelEncoder)
- Model training using XGBoost
- Evaluation using accuracy, F1-score, confusion matrix
- Feature importance analysis for business insights

## 📊 Key Features Driving Churn
- Short tenure
- Monthly contracts
- Electronic check payment
- Lack of online security or tech support

## 📁 Dataset

[Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## 💻 Tools Used

- Python (pandas, sklearn, xgboost, seaborn, matplotlib)
- Google Colab
- GitHub 
= Streamlit

## 🚀 Results

- Accuracy: ~80%
- Identified key churn triggers
- Helped frame data-backed retention strategy

## 📂 Run it Yourself

```bash
pip install -r requirements.txt
jupyter notebook churn_prediction.ipynb
