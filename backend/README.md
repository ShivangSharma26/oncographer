# Oncographer

Cancer Detection from Histopathology Slides using Patch-based Deep Learning.

## Overview
Oncographer is a deep-learning system that detects cancer in histopathology images. It uses a patch-based approach (PatchCamelyon dataset style) and is built using PyTorch, MLflow for tracking, Grad-CAM for explainability, and FastAPI for serving.

## Quick Start (Local Mock Testing)

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate Mock Data**
   Since the full PCam dataset is large, we have a script to generate a small mock dataset locally to test the training and inference pipeline:
   ```bash
   python generate_mock_data.py
   ```

3. **Train the Model**
   Run the training loop. This will use the models defined in `models/` and log metrics using MLflow.
   ```bash
   python train/train.py
   ```

4. **Serve the API**
   Once training completes (it will save a `best_model.pth`), start the FastAPI server:
   ```bash
   uvicorn serve.api:app --reload
   ```

5. **Test the API**
   You can go to `http://127.0.0.1:8000/docs` to test the `/predict` endpoint by uploading an image. The API returns a prediction probability and a Grad-CAM heatmap base64 string.
   
## Architecture
- `data/` - Dataset loading and augmentation pipelines.
- `models/` - Baseline CNN and ResNet Transfer learning architectures.
- `train/` - PyTorch training loop with MLflow integration.
- `eval/` - Evaluation metrics (Recall, Precision, AUC-ROC).
- `explain/` - Grad-CAM to show where the model focused.
- `serve/` - FastAPI implementation to accept an image patch and return prediction + heatmap.
