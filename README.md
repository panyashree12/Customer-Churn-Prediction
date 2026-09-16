# Customer Churn Prediction & Retention Analytics

## Project Overview

Customer churn prediction is an important business analytics problem that helps organizations identify customers who are likely to stop using their services.

This project develops a machine learning system to predict customer churn using a telecom customer dataset. Exploratory Data Analysis (EDA), feature engineering, classification algorithms, and customer risk analysis are used to understand churn behavior and support customer retention strategies.

## Objectives

- Analyze customer data and identify churn patterns.
- Perform data cleaning and exploratory data analysis.
- Engineer useful customer-related features.
- Build classification models for churn prediction.
- Compare Logistic Regression, Random Forest, and XGBoost.
- Evaluate models using Accuracy, Recall, F1-Score, and ROC-AUC.
- Identify customers with high churn risk.
- Provide an interactive Streamlit dashboard for analysis and prediction.

## Dataset

The project uses the Telco Customer Churn dataset.

- Number of customers: 7,043
- Original features: 21
- Target variable: Churn
- Churn classes: Yes / No

The dataset is stored in the `data` folder.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Joblib
- Streamlit
- Jupyter Notebook

## Project Workflow

1. Data Collection
2. Data Cleaning
3. Exploratory Data Analysis
4. Feature Engineering
5. Data Preprocessing
6. Model Training
7. Model Evaluation
8. Customer Risk Prediction
9. Streamlit Dashboard

## Machine Learning Models

### Logistic Regression

Used as a baseline classification model for predicting customer churn.

### Random Forest

An ensemble learning algorithm that combines multiple decision trees.

### XGBoost

A gradient boosting algorithm used for classification and churn prediction.

## Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 79.49% | 63.93% | 52.14% | 57.44% | 84.29% |
| Random Forest | 76.58% | 55.16% | 62.83% | 58.75% | 81.81% |
| XGBoost | 80.13% | 65.77% | 52.41% | 58.33% | 84.45% |

XGBoost achieved the highest ROC-AUC among the three models in this experiment.

Random Forest achieved the highest recall and F1-score for the churn class.

## Customer Risk Prediction

The project generates churn probabilities for customers and categorizes them into:

- Low Risk
- Medium Risk
- High Risk

This helps identify customers who may require retention attention.

## Streamlit Dashboard

An interactive Streamlit dashboard was developed with the following sections:

- Executive Overview
- Churn Analysis
- Customer Risk Analysis
- Model Performance
- Individual Prediction
- Business Recommendations

The dashboard allows users to explore churn patterns and generate individual customer churn predictions.

## Project Structure

```text
Customer-Churn-Prediction/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── models/
│   ├── churn_model.pkl
│   ├── customer_risk_predictions.csv
│   └── model_comparison.csv
│
├── notebooks/
│   └── customer_churn_analysis.ipynb
│
├── screenshots/
│
├── src/
│   └── train_model.py
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md