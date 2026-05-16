import pandas as pd
import numpy as np
import joblib
import json
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# MLflow is optional — training works with or without it
try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

# Support running from project root (python src/train.py) and from src/ directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from pipeline import clean_data, engineer_features

def main():
    print("Starting Training Pipeline...")
    
    if HAS_MLFLOW:
        mlflow.set_experiment("Customer_Churn_Prediction")
        print("  MLflow tracking enabled.")
    else:
        print("  MLflow not installed — skipping experiment tracking.")
    
    # 1. Load Data
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df = clean_data(df)
    df = engineer_features(df)
    
    # 2. Preprocess & Encode
    df_enc = pd.get_dummies(df, drop_first=True)
    
    X = df_enc.drop("Churn", axis=1)
    y = df_enc["Churn"]
    
    # Save feature columns for app
    feature_cols = list(X.columns)
    with open("feature_cols.json", "w") as f:
        json.dump(feature_cols, f)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale Data
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, "scaler.pkl")

    # KMeans (for clustering features in app)
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans.fit(X_train_scaled)
    joblib.dump(kmeans, "kmeans.pkl")
    
    # Train Classifier
    params = {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42
    }
    
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
    
    # Save model
    joblib.dump(model, "model.pkl")
    
    # Log to MLflow if available
    if HAS_MLFLOW:
        with mlflow.start_run():
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.log_artifact("scaler.pkl")
            mlflow.log_artifact("kmeans.pkl")
            mlflow.sklearn.log_model(model, "random_forest_model")
    
    print(f"Training complete. Metrics: {metrics}")
    print("Artifacts saved: model.pkl, scaler.pkl, kmeans.pkl, feature_cols.json")

if __name__ == "__main__":
    main()
