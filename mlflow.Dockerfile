FROM python:3.11-slim
WORKDIR /app

# Install MLflow
RUN pip install --no-cache-dir mlflow

# Copy the tracking database and mlruns
COPY mlflow.db ./
COPY backend/mlruns/ ./backend/mlruns/

EXPOSE 5000

# Start MLflow UI with 1 worker to prevent OOM, and bind to Render's injected $PORT
CMD ["sh", "-c", "mlflow ui --host 0.0.0.0 --port ${PORT:-5000} --workers 1"]
