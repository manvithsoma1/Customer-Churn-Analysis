#!/bin/bash
# Replace 'my-churn-dvc-bucket' with your actual AWS S3 bucket name
S3_BUCKET_NAME="my-churn-dvc-bucket"

echo "Configuring DVC remote to S3 bucket: $S3_BUCKET_NAME"

dvc init
dvc remote add -d myremote s3://$S3_BUCKET_NAME/dvcstore
dvc remote modify myremote endpointurl https://s3.amazonaws.com

echo "DVC remote configured."
echo "Note: Ensure you have your AWS credentials set in your environment variables or ~/.aws/credentials."
