import numpy as np
import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch import nn, optim

from model import ClimateDataset
from model import NeuralNet

def main():
    samples = np.load('data/samples_300_meteo_ex_norm.npy')
    # Extract features and target
    X = samples[0:2,:].T
    y = samples[2,:]

    # Split data into training and validation sets
    # Convert data to PyTorch tensors
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # Ensure correct shape
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    
    # Create DataLoaders
    batch_size = 64  
    train_loader = DataLoader(ClimateDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ClimateDataset(X_val_tensor, y_val_tensor), batch_size=batch_size, shuffle=False)

    
    # Initialize model, loss function, and optimizer
    input_dim = 2  # Number of predictors (ssrd and vpd)
    model = NeuralNet(input_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Starting training...",flush=True)
    # Training loop
    num_epochs = 1
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        # Validation Step
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                total_val_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.10f}, "
                f"Val Loss: {avg_val_loss:.10f}")

    torch.save(model.state_dict(), "./outputs/model_weights_default_meteo_ex_sbatch_test.pth")

if __name__ == "__main__":
    main()