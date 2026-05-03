#!/bin/bash
# Replace with your actual GitHub username and Repo Name
GITHUB_USER="manvithsoma1"
REPO_NAME="Customer-Churn-Analysis"
DAGSHUB_URL="https://dagshub.com/$GITHUB_USER/$REPO_NAME.dvc"

echo "Configuring DVC remote to DagsHub: $DAGSHUB_URL"

dvc remote add -d origin $DAGSHUB_URL
dvc remote modify origin --local auth basic
dvc remote modify origin --local user $GITHUB_USER
# You will be prompted for your DagsHub token as the password when pushing.

echo "DVC remote configured for DagsHub."
