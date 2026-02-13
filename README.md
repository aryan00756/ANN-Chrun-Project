# 🚀 Customer Churn Prediction using ANN (Streamlit Deployment)

## 📌 Overview

This project predicts customer churn using an Artificial Neural Network (ANN) built with TensorFlow/Keras.  
The trained model is deployed as an interactive web application using Streamlit.

The goal is to identify customers who are likely to leave the bank, helping businesses take preventive actions.

---

## 🧠 Tech Stack

- Python
- NumPy & Pandas
- Scikit-learn
- TensorFlow / Keras
- Streamlit

---

## 📂 Dataset

**Dataset:** Bank Customer Churn Dataset  

**Target Variable:**  
- `Exited` → 1 (Churned)  
- `Exited` → 0 (Stayed)

**Features Used:**
- CreditScore
- Geography
- Gender
- Age
- Tenure
- Balance
- NumOfProducts
- HasCrCard
- IsActiveMember
- EstimatedSalary

---

## ⚙️ Data Preprocessing

- Removed unnecessary columns (RowNumber, CustomerId, Surname)
- Label Encoding for Gender
- One-Hot Encoding for Geography
- Feature Scaling using StandardScaler
- Train-Test Split (80-20)

---
