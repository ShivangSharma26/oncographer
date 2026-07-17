FROM python:3.11-slim
WORKDIR /app

# Install MLflow
RUN pip install --no-cache-dir mlflow

# Copy the tracking database and mlruns
COPY mlflow.db ./
COPY backend/mlruns/ ./backend/mlruns/

# Start MLflow Server with 1 worker and bypass all security/CORS checks
CMD mlflow server --host 0.0.0.0 --port ${PORT:-10000} --workers 1 --backend-store-uri sqlite:///mlflow.db --serve-artifacts --default-artifact-root ./backend/mlruns --allowed-hosts '*'
