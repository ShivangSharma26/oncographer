import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
import sys

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.prepare import get_dataloaders
from data.augment import get_transforms
from models.baseline_cnn import BaselineCNN
from models.resnet_transfer import ResNetTransfer
from eval.evaluate import evaluate_model

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_model(config):
    model_type = config['model']['type']
    num_classes = config['model']['num_classes']
    
    if model_type == "baseline_cnn":
        return BaselineCNN(num_classes=num_classes)
    elif model_type == "resnet_transfer":
        return ResNetTransfer(num_classes=num_classes, fine_tune=True)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def train():
    config = load_config()
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set up data
    train_transform, val_transform = get_transforms(config['data']['image_size'])
    train_loader, val_loader, test_loader = get_dataloaders(config, train_transform, val_transform)

    # Set up model
    model = get_model(config).to(device)

    # Set up loss and optimizer
    # We use BCEWithLogitsLoss because our models output raw logits (no sigmoid at the end)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), 
                           lr=config['training']['learning_rate'], 
                           weight_decay=config['training']['weight_decay'])

    epochs = config['training']['epochs']
    
    # Use SQLite backend to avoid deprecated file store issues in MLflow 3.x
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(config['mlflow']['experiment_name'])
    
    with mlflow.start_run():
        # Log params
        mlflow.log_params(config['training'])
        mlflow.log_param("model_type", config['model']['type'])
        
        best_val_auc = 0.0

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(train_loader):
                inputs = inputs.to(device)
                labels = labels.to(device).view(-1, 1)

                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_train_loss = running_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f}")
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)

            # Validation
            val_metrics = evaluate_model(model, val_loader, device, threshold=config['training']['threshold'])
            
            mlflow.log_metric("val_auc", val_metrics['auc'], step=epoch)
            mlflow.log_metric("val_recall", val_metrics['recall'], step=epoch)
            mlflow.log_metric("val_precision", val_metrics['precision'], step=epoch)

            # Save best model
            if val_metrics['auc'] > best_val_auc:
                best_val_auc = val_metrics['auc']
                torch.save(model.state_dict(), "best_model.pth")
                print("-> Saved new best model based on AUC")

        print("Training complete. Best Val AUC:", best_val_auc)
        
        # Test Evaluation
        print("Loading best model for test evaluation...")
        model.load_state_dict(torch.load("best_model.pth"))
        test_metrics = evaluate_model(model, test_loader, device, threshold=config['training']['threshold'])
        
        mlflow.log_metric("test_auc", test_metrics['auc'])
        mlflow.log_metric("test_recall", test_metrics['recall'])
        mlflow.log_metric("test_precision", test_metrics['precision'])
        
        # Log model to MLflow with an input example to fix PyTorch 2 serialization
        example_input = next(iter(train_loader))[0].numpy()
        mlflow.pytorch.log_model(model, "model", input_example=example_input)

if __name__ == "__main__":
    train()
