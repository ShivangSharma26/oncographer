# Oncographer — Deep Learning Pathology Patch Classifier

An intelligent, end-to-end medical AI platform designed to detect malignancy in histopathology tissue patches with high accuracy and explainability.

Built to solve real-world clinical diagnostic challenges by combining the powerful feature-extraction capabilities of a fine-tuned **ResNet18** model with the visual interpretability of **Grad-CAM heatmaps**.

### 🎬 Live Demonstrations
📂 **[View the Live Application](https://oncographer.onrender.com)**  
📂 **[View the MLflow Training Dashboard](https://oncographer-mlflow.onrender.com)** *(If deployed separately)*

| Feature | Description |
|---|---|
| **REAL-TIME INFERENCE** | A complete web-based interface where pathologists can drag-and-drop tissue patches and receive instant cancer probability scores. |
| **EXPLAINABLE AI (XAI)** | Live generation of Grad-CAM heatmaps overlaid on the original tissue sample, highlighting the exact cellular structures the AI used to make its decision. |

---

## 🎯 Project Goal

The primary objective of this project is to build a robust, clinical-grade AI screening tool capable of handling core diagnostic queries:
- **Malignancy Detection:** Accurately classifying 224x224 histopathology patches as Benign or Malignant.
- **Clinical Interpretability:** Providing visual evidence (heatmaps) to doctors, bridging the gap between "black-box" AI and clinical trust.
- **Model Tracking:** Systematically logging every training run, hyperparameter, and validation metric to ensure medical compliance and reproducibility.

The system relies on a PyTorch-based pipeline utilizing **Transfer Learning** on the PatchCamelyon dataset, optimized for high Recall to minimize false negatives in cancer detection.

---

## 🏗️ System Architecture & High-Level Design

Oncographer relies on a decoupled architecture built for scalability and clinical deployment:

- **Frontend Client (React/Vite):** A premium, glassmorphism-styled UI built with TailwindCSS v4. It handles client-side image validation and asynchronous API communication.
- **Inference Server (FastAPI):** A high-performance Python backend that ingests base64 images, runs the normalization pipeline, and performs PyTorch tensor inference.
- **Explainability Engine (Grad-CAM):** Intercepts the gradients from the final convolutional layer of ResNet18 to generate a spatial activation map.
- **MLOps Tracking (MLflow):** A dedicated SQLite-backed MLflow instance tracking all experiments, loss curves, and artifact logging.

### Bonus Features Implemented
- ✅ **Dockerized Infrastructure:** A highly reproducible multi-stage `Dockerfile` that builds the React frontend and serves it dynamically via FastAPI in a single container.
- ✅ **Explainable AI:** Custom Grad-CAM implementation bypassing standard library limitations to guarantee exact overlay registration using OpenCV.
- ✅ **Production-Ready Routing:** Integrated static file serving alongside REST API endpoints to bypass CORS overhead in production.

---

## 💻 Tech Stack

**Core Pipeline & AI**
- Framework: Python, PyTorch, Torchvision
- Computer Vision: OpenCV, Pillow, Albumentations
- MLOps: MLflow

**Backend & API**
- Framework: FastAPI, Uvicorn

**Frontend**
- Framework: React, Vite
- Styling: TailwindCSS v4 (PostCSS)

**Infrastructure**
- Docker (Multi-stage builds)
- Render (Cloud Deployment)

---

## 📂 Codebase & Folder Structure

The repository is highly organized for code clarity, making it easy for any engineer or researcher to scale.

```text
oncographer/
├── backend/                  # Python API and ML Pipeline
│   ├── data/                 # Augmentation and DataLoader logic
│   ├── eval/                 # Evaluation scripts (AUC-ROC, Recall)
│   ├── explain/              # Grad-CAM XAI implementation
│   ├── models/               # ResNet18 architecture definitions
│   ├── serve/                # FastAPI entry point (api.py)
│   └── train/                # Training loop and MLflow logging
├── frontend/                 # React User Interface
│   ├── src/                  # Components, Hooks, and API logic
│   ├── index.css             # Tailwind v4 configuration
│   └── vite.config.js        # Build tooling
├── mlflow.db                 # SQLite database for experiment tracking
├── Dockerfile                # Multi-stage build for unified deployment
├── mlflow.Dockerfile         # Isolated deployment for MLflow dashboard
└── README.md                 # You are here
```

---

## 🚀 Setup Instructions

### 1. Prerequisites
- Python 3.11+
- Node.js 20+

### 2. Backend Setup
Create and activate a virtual environment:

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:
```bash
cd backend
pip install -r requirements.txt
```

### 3. Frontend Setup
```bash
cd frontend
npm install
npm run build
```

### 4. Start the Servers
To run the MLflow Dashboard:
```bash
python -m mlflow ui
```

To run the FastAPI Server (which also serves the frontend):
```bash
cd backend
python -m uvicorn serve.api:app --reload
```
The application will start on `http://localhost:8000`.

---

## 🌍 Deployment

Oncographer's architecture is designed for easy 1-click deployment to cloud platforms:

- **Unified Application:** Deploy the primary `Dockerfile` to Render, Hugging Face Spaces, or AWS App Runner. It builds the UI and serves the API concurrently on port 7860.
- **MLOps Dashboard:** Deploy the `mlflow.Dockerfile` as a secondary internal service to expose the tracking dashboard securely.

---

© Copyright 2026. Made with love by Shivang Sharma.
