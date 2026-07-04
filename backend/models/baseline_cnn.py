import torch.nn as nn
import torch.nn.functional as F

class BaselineCNN(nn.Module):
    def __init__(self, num_classes=1):
        super(BaselineCNN, self).__init__()
        # Simple CNN architecture to serve as a baseline to beat.
        # Input: 3 x 96 x 96
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # 16 x 48 x 48
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # 32 x 24 x 24
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # 64 x 12 x 12
        
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 12 * 12) # flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
