import torch
import torch.nn as nn


# Discriminator network class
class Discriminator(nn.Module):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size  # Batch size for input data
        self.fc1 = nn.Linear(1 * 28 * 28, 128)  # Fully connected layer 1
        self.LReLu = nn.LeakyReLU()  # Leaky ReLU activation function
        self.fc2 = nn.Linear(128, 1)  # Fully connected layer 2
        self.SigmoidL = nn.Sigmoid()  # Sigmoid activation function

    # Function for forward propagation
    def forward(self, x, y):
        flat_x = x.view(self.batch_size, -1)  # Flatten the input image
        flat_y = y.view(self.batch_size, -1)
        flat = flat_x - flat_y
        # print the shapes of flat_x, flat_y, and flat
        print(f"flat_x shape: {flat_x.shape}, flat_y shape: {flat_y.shape}, flat shape: {flat.shape}")
        layer1 = self.LReLu(self.fc1(flat))  # Apply Leaky ReLU to the first fully connected layer
        out = self.SigmoidL(self.fc2(layer1))  # Apply Sigmoid to the second fully connected layer
        return out.view(-1, 1).squeeze(1)  # Flatten the output and remove unnecessary dimension