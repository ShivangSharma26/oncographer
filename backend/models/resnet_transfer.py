import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ResNetTransfer(nn.Module):
    def __init__(self, num_classes=1, fine_tune=True):
        super(ResNetTransfer, self).__init__()
        # Load a pre-trained ResNet model
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # If we don't want to fine-tune the whole model, we can freeze the early layers
        if not fine_tune:
            for param in self.model.parameters():
                param.requires_grad = False
                
        # Replace the final fully connected layer for binary classification
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)
