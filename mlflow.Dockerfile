FROM python:3.11-slim
WORKDIR /app

# Install MLflow
RUN pip install --no-cache-dir mlflow

# Copy the tracking database and mlruns
COPY mlflow.db ./
COPY backend/mlruns/ ./backend/mlruns/

EXPOSE 5000

# Start MLflow UI with only 1 worker to prevent Out-Of-Memory (OOM) on free tiers
CMD ["sh", "-c", "mlflow ui --host 0.0.0.0 --port 5000 --gunicorn-opts '--workers 1'"]
