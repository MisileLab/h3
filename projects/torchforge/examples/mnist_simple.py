"""
Simple MNIST CNN example using TorchForge generated code.

This is an example of the code that TorchForge generates for a simple CNN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.relu = nn.ReLU()
        self.maxpool2d = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=32 * 13 * 13, out_features=128)
        self.relu_1 = nn.ReLU()
        self.linear_1 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.relu(x)
        x = self.maxpool2d(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.relu_1(x)
        x = self.linear_1(x)
        return x


# Example usage:
if __name__ == "__main__":
    model = SimpleMNISTCNN()
    print(model)
    
    # Test with dummy input
    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f'Output shape: {output.shape}')
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')