import torch
import numpy as np
from sklearn.metrics import roc_auc_score, recall_score, precision_score, classification_report

def evaluate_model(model, dataloader, device, threshold=0.5):
    """
    Evaluates the model on the given dataloader.
    Returns standard medical metrics: Recall, Precision, AUC.
    """
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device).view(-1, 1)

            outputs = model(inputs)
            # Apply sigmoid since model outputs raw logits
            probs = torch.sigmoid(outputs)
            
            # Use specific threshold to classify
            preds = (probs >= threshold).float()

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_labels = np.array(all_labels).flatten()
    all_probs = np.array(all_probs).flatten()
    all_preds = np.array(all_preds).flatten()

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        # Occurs if only one class is present in the batch/dataset
        auc = 0.5
        
    recall = recall_score(all_labels, all_preds, zero_division=0)
    precision = precision_score(all_labels, all_preds, zero_division=0)

    print("\n--- Evaluation Results ---")
    print(f"Threshold: {threshold}")
    print(f"AUC-ROC:   {auc:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, zero_division=0, target_names=["Healthy", "Cancer"]))

    return {"auc": auc, "recall": recall, "precision": precision}
