import torch.nn as nn
from torch.utils.data import Dataset

# Define the neural network model
class NeuralNet(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Output layer, no activation
        )

    def forward(self, x):
        return self.model(x)
    

# Define the dataset class
class ClimateDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x_i = self.X[idx]
        y_i = self.y[idx]
        return x_i, y_i