import pandas as pd
import numpy as np
import mlflow
import joblib
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from pipeline import clean_data, engineer_features

def main():
    print("🚀 Starting MLflow Training Pipeline...")
    
    # Configure MLflow
    mlflow.set_experiment("Customer_Churn_Prediction")
    
    # 1. Load Data
    df = pd.read_csv("../WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df = clean_data(df)
    df = engineer_features(df)
    
    # 2. Preprocess & Encode
    df_enc = pd.get_dummies(df, drop_first=True)
    
    X = df_enc.drop("Churn", axis=1)
    y = df_enc["Churn"]
    
    # Save feature columns for app
    feature_cols = list(X.columns)
    with open("../feature_cols.json", "w") as f:
        json.dump(feature_cols, f)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    with mlflow.start_run():
        # Scale Data
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        joblib.dump(scaler, "../scaler.pkl")
        mlflow.log_artifact("../scaler.pkl")

        # KMeans (for clustering features in app)
        kmeans = KMeans(n_clusters=4, random_state=42)
        kmeans.fit(X_train_scaled)
        joblib.dump(kmeans, "../kmeans.pkl")
        mlflow.log_artifact("../kmeans.pkl")
        
        # Train Classifier
        params = {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42
        }
        mlflow.log_params(params)
        
        model = RandomForestClassifier(**params)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred)
        }
        mlflow.log_metrics(metrics)
        
        # Save & Log Model
        joblib.dump(model, "../model.pkl")
        mlflow.sklearn.log_model(model, "random_forest_model")
        
        print("✅ Training complete. Artifacts saved and logged to MLflow.")

if __name__ == "__main__":
    main()
