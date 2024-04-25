from lightgbm import train
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import time


# Function to get the number of features from the dataloader
def get_num_features(dataloader):
    # Iterate over dataloader to get the first batch
    for data, _ in dataloader:
        return data.shape[1]  # Return the second dimension (num_features)


class SimpleNN(nn.Module):
    def __init__(self, size=466):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Sigmoid activation for binary classification
        return x


# def train_SimpleNN(X_train, y_train, epochs=100):
#     device = check_device()
#     # Initialize the model, loss function, and optimizer
#     model = SimpleNN()
#     # move model to device
#     model.to(device)
#     criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for binary classification

#     optimizer = optim.Adam(model.parameters(), lr=0.001)

#     X_train = torch.tensor(X_train).float().to(device)
#     y_train = torch.tensor(y_train).float().to(device)

#     # Training loop
#     for epoch in range(epochs):  # Loop over the dataset multiple times
#         optimizer.zero_grad()  # Zero the parameter gradients
#         outputs = model(X_train)  # Forward pass
#         loss = criterion(outputs, y_train)  # Calculate loss
#         loss.backward()  # Backward pass
#         optimizer.step()  # Optimize
#         print(f"epoch {epoch} of {epochs} epochs")


def train_SimpleNN(train_loader, test_loader, epochs=100):
    device = check_device()
    train_features = get_num_features(train_loader)
    validation_features = get_num_features(test_loader)
    print(f"Train_features: {train_features}")
    assert (
        train_features == validation_features
    ), f"train features = {train_features} must be equal to validation features = {validation_features}"
    model = SimpleNN(size=train_features)  # TODO: do this automatically.
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # To store training and validation loss for monitoring
    training_loss = []
    validation_loss = []
    start_time = time.time()
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        total_train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        # Calculate average training loss per epoch
        avg_train_loss = total_train_loss / len(train_loader)
        training_loss.append(avg_train_loss)

        # Evaluation on the test set
        model.eval()  # Set model to evaluation mode
        total_val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                total_val_loss += loss.item()

        # Calculate average validation loss per epoch
        avg_val_loss = total_val_loss / len(test_loader)
        validation_loss.append(avg_val_loss)
        epoch_duration = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f} finished in {epoch_duration:.2f} seconds"
        )

    return training_loss, validation_loss


# def data_split(df, output):
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(df)
#     # X_scaled = df

#     # Splitting the data into training and testing sets
#     # Using a test size of 20% and a random state for reproducibility
#     X_train, X_test, y_train, y_test = train_test_split(X_scaled, output, test_size=0.2, random_state=42)

#     # Displaying the sizes of the splits to verify
#     print(f"Training data shape: {X_train.shape}")
#     print(f"Testing data shape: {X_test.shape}")
#     # Convert data to tensors
#     train_dataset = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train.values).float())
#     test_dataset = TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test.values).float())
#     # Creating data loaders
#     batch_size = 64
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#     # X_train = torch.randn(100, 500)  # 100 samples, 500 features -> directly train the joined table
#     # y_train = torch.randint(0, 2, (100, 1)).type(torch.FloatTensor)  # Binary labels (0 or 1)
#     return X_train, y_train#, train_loader, test_loader
#     # train_model(train_loader, criterion, optimizer)
# def data_split(df, output):
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(df)

#     # Splitting the data into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(X_scaled, output, test_size=0.2, random_state=42)

#     # Ensure that y_train and y_test are numpy arrays, not DataFrames
#     y_train_array = y_train.values.to_numpy()  # This flattens the array if y_train is a DataFrame column
#     y_test_array = y_test.values.to_numpy()    # Similarly for y_test

#     # Convert data to tensors
#     train_dataset = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train_array).float())
#     test_dataset = TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test_array).float())

#     # Creating data loaders
#     batch_size = 64
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#     return train_loader, test_loader


def data_split(df_in, output_in):
    # Identify numeric columns (assuming non-numeric are object types or datetime)
    df = df_in.select_dtypes(include=["number"])
    # pd.DataFrame.select_dtypes
    # df = df.values
    # Scale only numeric columns
    scaler = StandardScaler()
    # df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    dfvals = df.values
    output = output_in.values
    df_scaled = scaler.fit_transform(dfvals)

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        df_scaled, output, test_size=0.2, random_state=42
    )

    # Ensure that y_train and y_test are numpy arrays
    y_train_array = y_train
    y_test_array = y_test
    # y_test_array = y_test.values.ravel()
    # Convert data to tensors
    train_dataset = TensorDataset(
        torch.tensor(X_train).float(), torch.tensor(y_train_array).float()
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test).float(), torch.tensor(y_test_array).float()
    )

    # Creating data loaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


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
    df = pd.DataFrame(data, columns=["a", "b", "c", "d"])
    df["e"] = labels
    print(df)

    X = df[["a", "b", "c", "d"]]
    y = df["e"]  # Assuming 'e' is binary for classification

    X_train, y_train = data_split(X, y)

    print("training")
    train_SimpleNN(X_train, y_train)
    print("DONE")
