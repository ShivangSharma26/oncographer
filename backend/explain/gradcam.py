import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        """
        Args:
            model: PyTorch model.
            target_layer: The layer to compute the gradients and activations for.
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def generate_heatmap(self, input_image, class_idx=None):
        """
        Generates the Grad-CAM heatmap.
        Args:
            input_image: Tensor of shape (1, C, H, W).
            class_idx: Int (0 for healthy, 1 for cancer). If None, uses model's prediction.
        Returns:
            heatmap (np.ndarray): 2D numpy array of the heatmap.
            prediction (float): Probability of the positive class.
        """
        self.model.eval()
        
        # Forward pass
        model_output = self.model(input_image)
        prob = torch.sigmoid(model_output).item()
        
        # We only have one output node for binary classification
        # The backward pass needs to be computed on the logits
        self.model.zero_grad()
        model_output.backward()
        
        # Get gradients and activations
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        # Global average pooling on gradients
        weights = np.mean(gradients, axis=(1, 2))
        
        # Weight activations
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        # Apply ReLU to only keep positive influence
        cam = np.maximum(cam, 0)
        
        # Normalize between 0 and 1
        cam = cv2.resize(cam, (input_image.shape[3], input_image.shape[2]))
        if np.max(cam) != 0:
            cam = cam - np.min(cam)
            cam = cam / np.max(cam)
            
        return cam, prob

def overlay_heatmap(image_tensor, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    Overlays the heatmap on the original image.
    Args:
        image_tensor: Normalized image tensor (1, C, H, W).
        heatmap: 2D numpy array.
    Returns:
        overlaid_img: RGB numpy image.
    """
    # Denormalize image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    img = image_tensor[0].cpu().numpy().transpose(1, 2, 0)
    img = std * img + mean
    img = np.clip(img, 0, 1)
    img = np.uint8(255 * img)
    
    # Convert heatmap to RGB using colormap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Overlay
    superimposed_img = heatmap_colored * alpha + img * (1 - alpha)
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    
    return superimposed_img
