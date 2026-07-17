FROM python:3.11-slim
WORKDIR /app

# Install MLflow
RUN pip install --no-cache-dir mlflow

# Copy the tracking database and mlruns
COPY mlflow.db ./
COPY backend/mlruns/ ./backend/mlruns/

# Start MLflow UI with 1 worker to prevent OOM, bind to Render's injected $PORT, and allow external hosts
CMD ["sh", "-c", "mlflow ui --host 0.0.0.0 --port ${PORT:-5000} --workers 1 --allowed-hosts '*'"]
