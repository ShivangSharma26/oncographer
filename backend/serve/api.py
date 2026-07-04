from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import torch
import io
import base64
from PIL import Image
import os
import sys
import yaml

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.augment import get_transforms
from models.resnet_transfer import ResNetTransfer
from explain.gradcam import GradCAM, overlay_heatmap

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Oncographer API", description="API for cancer detection from pathology patches")

# Enable CORS for the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load configuration and model globally
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
model = ResNetTransfer(num_classes=config['model']['num_classes'], fine_tune=True)

# In a real scenario, handle missing best_model.pth gracefully
if os.path.exists("best_model.pth"):
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

# Initialize Grad-CAM targeting the last convolutional layer of ResNet
target_layer = model.model.layer4[-1]
gradcam = GradCAM(model, target_layer)

_, val_transform = get_transforms(config['data']['image_size'])

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

@app.get("/")
def serve_react_app():
    # Serve the React index.html for the root route
    react_build_path = os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "dist")
    index_path = os.path.join(react_build_path, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"status": "ok", "message": "Oncographer API is running, but frontend build is missing."}

# We will mount static files at the very bottom

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Only image files are allowed.")
    
    try:
        contents = await file.read()
        print(f"DEBUG: Read {len(contents)} bytes from file")
        image = Image.open(io.BytesIO(contents)).convert('RGB')
    except Exception as e:
        print(f"DEBUG: Exception in reading image: {repr(e)}")
        raise HTTPException(status_code=400, detail=f"Error reading image: {repr(e)}")
        
    # Preprocess
    input_tensor = val_transform(image).unsqueeze(0).to(device)
    
    # Generate Heatmap and Prediction
    heatmap, prob = gradcam.generate_heatmap(input_tensor)
    
    # Overlay heatmap for visualization
    overlaid_img = overlay_heatmap(input_tensor, heatmap)
    
    # Convert image back to base64 for API response
    result_img = Image.fromarray(overlaid_img)
    buffered = io.BytesIO()
    result_img.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return JSONResponse(content={
        "cancer_probability": round(prob, 4),
        "is_suspicious": bool(prob >= config['training']['threshold']),
        "gradcam_heatmap_base64": img_base64
    })

# Mount the static assets (js, css, images) from the React build
# MUST BE AT THE BOTTOM so it doesn't intercept API routes
react_build_path = os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "dist")
if os.path.exists(react_build_path):
    app.mount("/", StaticFiles(directory=react_build_path), name="static")
