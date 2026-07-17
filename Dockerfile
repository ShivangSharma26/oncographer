# Stage 1: Build the React frontend
FROM node:20 as build-stage
WORKDIR /app/frontend

# Copy frontend source and install dependencies
COPY frontend/package*.json ./
RUN npm install

# Build the frontend
COPY frontend/ ./
RUN npm run build

# Stage 2: Serve the API and frontend using Python
FROM python:3.11-slim
WORKDIR /app

# Install system dependencies required for OpenCV and PyTorch
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU version explicitly to save massive RAM/Disk space on free tiers
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining backend dependencies (ensure torch is removed or satisfied)
COPY backend/requirements.txt ./backend/
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy the backend source code
COPY backend/ ./backend/

# Copy the built React app from the build-stage
COPY --from=build-stage /app/frontend/dist ./frontend/dist

# Expose port 7860 for Hugging Face Spaces
EXPOSE 7860

# Set the working directory to backend where api.py and config are located
WORKDIR /app/backend

# Start the FastAPI server using Uvicorn
CMD ["uvicorn", "serve.api:app", "--host", "0.0.0.0", "--port", "7860"]
