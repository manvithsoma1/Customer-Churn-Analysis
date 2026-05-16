FROM python:3.9-slim

WORKDIR /app

# Install requirements first (cached layer — only rebuilds if deps change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model artifacts
COPY src/ ./src/
COPY app.py .
COPY model.pkl .
COPY scaler.pkl .
COPY kmeans.pkl .
COPY feature_cols.json .
COPY WA_Fn-UseC_-Telco-Customer-Churn.csv .

# DVC files (needed for pipeline metadata, not runtime)
COPY dvc.yaml .
COPY dvc.lock .

# Expose Streamlit port
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", \
  "--server.port=8501", \
  "--server.address=0.0.0.0", \
  "--server.headless=true", \
  "--browser.gatherUsageStats=false"]
