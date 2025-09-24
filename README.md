# Telecom-Customer-Churn-Prediction-using-Machine-Learning
Developed a Random Forest model on telecom data to predict customer churn, achieving 86% accuracy with key insights into churn drivers.
📊 Telecom Customer Churn Prediction
📌 Project Overview

Customer churn is a major challenge in the telecom industry, where retaining existing customers is often more cost-effective than acquiring new ones. This project focuses on building a machine learning model to predict customer churn based on demographic, account, and service-related data.

By identifying customers likely to churn, telecom companies can take proactive retention measures.

🎯 Objectives

Analyze telecom customer dataset.

Build a machine learning model to predict churn.

Evaluate performance using standard metrics.

Provide business insights into churn drivers.

📂 Dataset

Source: https://drive.google.com/file/d/1klNBckeyAq8lflHIdwqRm95_obURo9Kg/view?usp=sharing
<img width="764" height="787" alt="image" src="https://github.com/user-attachments/assets/a8d00af9-aa63-4076-8aaa-079f4b1c9b15" />
<img width="756" height="645" alt="image" src="https://github.com/user-attachments/assets/91627ad4-8d33-4563-bc2d-8f4100869f0c" />



Features:

Customer demographics (e.g., gender, senior citizen)

Account information (e.g., tenure, contract type, payment method)

Service usage (e.g., internet service, phone service, streaming)

Target Variable: Churn (Yes/No)

🛠️ Technologies Used

Programming Language: Python

Libraries:

pandas, numpy → Data preprocessing

matplotlib, seaborn → Data visualization

scikit-learn → Random Forest model, evaluation

joblib → Model saving

streamlit (optional) → Frontend app

🔑 Steps in the Project

Data Preprocessing

Handled missing values

Encoded categorical variables

Feature scaling

Exploratory Data Analysis (EDA)

Visualized churn vs non-churn distribution

Analyzed correlation among features

Identified important predictors

Model Building

Implemented Random Forest Classifier

Tuned hyperparameters for better performance

Model Evaluation

Accuracy: 86%

Precision, Recall, F1-score

Confusion Matrix

Deployment (Optional)

Model saved using joblib

Streamlit app for user-friendly predictions

📊 Results

Best Model: Random Forest

Accuracy: 86%

Key factors influencing churn: contract type, tenure, monthly charges, and internet service.
