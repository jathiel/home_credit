import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(500, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Sigmoid activation for binary classification
        return x


def train_SimpleNN(X_train, y_train, epochs=100):
    device = check_device()
    # Initialize the model, loss function, and optimizer
    model = SimpleNN()
    # move model to device
    model.to(device)
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for binary classification

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Convert datasets to tensors if they're not already, and ensure the labels are float for BCELoss
    X_train = torch.tensor(X_train).float().to(device)
    y_train = torch.tensor(y_train).float().to(device)

    # Training loop
    for epoch in range(epochs):  # Loop over the dataset multiple times
        optimizer.zero_grad()  # Zero the parameter gradients
        outputs = model(X_train)  # Forward pass
        loss = criterion(outputs, y_train)  # Calculate loss
        loss.backward()  # Backward pass
        optimizer.step()  # Optimize
        print(f"epoch {epoch} of {epochs} epochs")


def data_split(df, output):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    # Splitting the data into training and testing sets
    # Using a test size of 20% and a random state for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, output, test_size=0.2, random_state=42)

    # Displaying the sizes of the splits to verify
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    # Convert data to tensors
    train_dataset = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train.values).float())
    test_dataset = TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test.values).float())
    # Creating data loaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Assuming X_train and y_train are your training data and labels, respectively
    X_train = torch.randn(100, 500)  # 100 samples, 500 features -> directly train the joined table
    y_train = torch.randint(0, 2, (100, 1)).type(torch.FloatTensor)  # Binary labels (0 or 1)
    return X_train, y_train#, train_loader, test_loader
    # train_model(train_loader, criterion, optimizer)


def check_device():
    device = None
    # Check if CUDA is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available. Using the GPU...")
    else:
        device = torch.device("cpu")
        print("GPU not available, defaulting to CPU instead...")
    return device


if __name__ == "__main__":

    np.random.seed(4269)
    num_rows = 999
    data = np.random.rand(num_rows, 4)  # This generates values between 0 and 1
    labels = np.random.randint(2, size=(num_rows, 1))  # Generates 0 or 1
    print(labels.shape)
    # Create DataFrame
    df = pd.DataFrame(data, columns=['a', 'b', 'c', 'd'])
    df['e'] = labels
    print(df)

    X = df[['a', 'b', 'c', 'd']]
    y = df['e']  # Assuming 'e' is binary for classification

    X_train, y_train= data_split(X, y)

    print("training")
    train_SimpleNN(X_train, y_train)
    print("DONE")
