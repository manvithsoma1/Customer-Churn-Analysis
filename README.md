# Customer Churn Prediction Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Accuracy](https://img.shields.io/badge/accuracy-83.42%25-brightgreen.svg)]()
[![AUC-ROC](https://img.shields.io/badge/AUC--ROC-0.9304-brightgreen.svg)]()

## Overview
This repository contains a complete, production-ready machine learning pipeline for predicting customer churn using the Telco Customer Churn dataset.

The project demonstrates:
- End-to-end Data Processing
- Exploratory Data Analysis (EDA)
- Advanced Feature Engineering
- Model Evaluation & Comparison (Logistic Regression, Random Forest, Gradient Boosting)
- Model Interpretability with SHAP
- Customer Segmentation using K-Means Clustering
- Actionable Business Insights and Revenue Impact Analysis

## Project Structure
- `notebooks/churn_analysis.ipynb`: The primary analysis notebook showcasing the end-to-end data science workflow.
- `src/pipeline.py`: Reusable data processing and feature engineering functions.
- `reports/insights.md`: Business recommendations and financial impact analysis.
- `reports/`: Folder containing generated visualization plots.

## Installation & Setup
1. Clone the repository
2. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook to explore the models and see the resulting performance.

## Results
- **Best Model:** Random Forest
- **Accuracy:** 83.42%
- **AUC-ROC:** 0.9304
- **Revenue at Risk:** $1,349,603.40 (Potential Savings: $269,920.68)

Read `reports/insights.md` for full business recommendations.
