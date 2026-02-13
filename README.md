🚀 Customer Churn Prediction using ANN (Streamlit Deployment)
📌 Project Overview

This project predicts whether a customer will churn (leave the bank) using an Artificial Neural Network (ANN) built with TensorFlow/Keras.

The model is deployed as an interactive web app using Streamlit, allowing users to input customer details and get real-time churn predictions.

🎯 Problem Statement

Customer churn directly impacts business revenue. Retaining customers is cheaper than acquiring new ones.

This project aims to:

Predict customer churn probability

Help businesses identify high-risk customers

Enable data-driven retention strategies

🧠 Tech Stack

Python

NumPy, Pandas

Scikit-learn

TensorFlow / Keras

Streamlit

Matplotlib / Seaborn (EDA)

📂 Dataset

Dataset used: Bank Customer Churn Dataset

Typical Features:

Credit Score

Geography

Gender

Age

Tenure

Balance

Number of Products

Has Credit Card

Is Active Member

Estimated Salary

Target Variable:

Exited → 1 (Churned), 0 (Stayed)

🔍 Data Preprocessing

Removed irrelevant columns (e.g., RowNumber, CustomerId)

One-Hot Encoding (Geography)

Label Encoding (Gender)

Feature Scaling using StandardScaler

Train-Test Split (80-20)

🧠 Model Architecture (ANN)

Input Layer

Hidden Layer 1 → Dense (ReLU)

Hidden Layer 2 → Dense (ReLU)

Output Layer → Sigmoid (Binary Classification)

Loss Function:

Binary Crossentropy

Optimizer:

Adam

Evaluation Metrics:

Accuracy

Confusion Matrix

📊 Model Performance

Accuracy: ~XX%

Good precision-recall balance

Reduced overfitting using proper scaling

(Replace with your actual numbers.)

🌐 Streamlit Deployment

The model is deployed using Streamlit for real-time prediction.

Features:

Interactive input sliders

Dropdowns for categorical values

Real-time churn probability

Clean UI

▶️ How to Run Locally
1️⃣ Clone the Repository
git clone https://github.com/yourusername/ann-churn-prediction.git
cd ann-churn-prediction

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run Streamlit App
streamlit run app.py

📁 Project Structure
├── app.py
├── model.h5
├── scaler.pkl
├── label_encoder.pkl
├── requirements.txt
├── churn_model.ipynb
└── README.md

🚀 Future Improvements

Hyperparameter tuning

Add dropout to prevent overfitting

Use advanced architectures (BatchNorm, EarlyStopping)

Deploy on AWS / Render / Streamlit Cloud

Add SHAP for model interpretability

📌 Why This Project Matters

This project demonstrates:

End-to-end ML pipeline

ANN implementation from scratch

Real-world business use case

Model deployment skills

Production-ready workflow
