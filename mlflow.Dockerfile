FROM python:3.11-slim
WORKDIR /app

# Install MLflow
RUN pip install --no-cache-dir mlflow

# Copy the tracking database and mlruns
COPY mlflow.db ./
COPY backend/mlruns/ ./backend/mlruns/

EXPOSE 5000

# Start MLflow UI
CMD ["mlflow", "ui", "--host", "0.0.0.0", "--port", "5000"]
