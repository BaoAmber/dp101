import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNet(nn.Module):
    """Simple fully connected neural network model for MNIST digit recognition"""
    
    def __init__(self):
        super(SimpleNet, self).__init__()
        # Input layer: 28*28 = 784 pixels
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        # Output layer: 10 digit classes (0-9)
        self.fc3 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        # Flatten input image
        x = x.view(-1, 28 * 28)
        # First layer + ReLU activation
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        # Second layer + ReLU activation
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        # Output layer
        x = self.fc3(x)
        return x

