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
from sklearn.metrics import roc_curve, auc, roc_auc_score


# Function to get the number of features from the dataloader
def get_num_features(dataloader):
    """Returns the number of features in a dataloader

    Args:
        dataloader (DataLoader): Load dataloader

    Returns:
        num_feaatures: number of features
    """
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

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.gru.num_layers, x.size(0), self.gru.hidden_size).to(x.device)
        
        # Forward propagate GRU
        out, _ = self.gru(x, h0)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


def train_SimpleNN(train_loader, test_loader, epochs=100):
    """Trains a simple neural network given training and validation dataloaders

    Args:
        train_loader (DataLoader): DataLoader for training data
        test_loader (DataLoader): DataLoader for validation data
        epochs (int, optional): Number of epochs. Defaults to 100.

    Returns:
        Train Losses: list of train losses
        Validation Losses: list of validation losses
    """
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
    # Initialize variables to track the best model
    best_val_loss = float('inf')
    best_epoch = 0

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
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), 'W:/Erdos/Project/home_credit/data/bestSimpleNN.pth')  # Save the model parameters
            print(f"Model saved at epoch {epoch+1} with Validation Loss: {avg_val_loss:.4f}")
        print(f"Best model was saved at epoch {best_epoch+1} with a validation loss of {best_val_loss:.4f}")

        print(
            f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f} finished in {epoch_duration:.2f} seconds"
        )

    return training_loss, validation_loss
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt

def evaluate_model(model, loader, plt_title=""):
    """Evaluate a trained neural network on a test set and compute ROC AUC score.

    Args:
        model (torch.nn.Module): The trained neural network model.
        test_loader (DataLoader): DataLoader for validation or test data.

    Returns:
        float: The average loss on the test set.
        list: The list of predicted probabilities.
        float: The ROC AUC score for the predictions.
    """
    device = check_device()
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    criterion = nn.BCELoss()
    total_loss = 0
    y_true = []
    y_scores = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
            y_true.extend(y_batch.cpu().numpy())
            y_scores.extend(outputs.cpu().numpy())  # Collect raw probabilities

    average_loss = total_loss / len(loader)
    
    # Calculate ROC AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    # ROC curve    
    fig =plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic : {plt_title}')
    plt.legend(loc="lower right")
    plt.show()
    fig.savefig(f"W:/Erdos/Project/home_credit/data{plt_title}.png")
    
    # Print out the AUC score
    print(f"ROC AUC Score: {roc_auc:.4f}")
    
    return average_loss, y_scores, roc_auc

# average_loss, test_probabilities, roc_auc_score = evaluate_model(trained_model, test_loader)

def data_split(df_in: pd.DataFrame, output_in: pd.DataFrame, return_dfs:bool=False, seed:int=42):
    """Split data into training and validation datasets or dataloaders

    Args:
        df_in (pd.DataFrame): Input Dataframe
        output_in (pd.DataFrame): Output column from Dataframe
        return_dfs (bool, optional): return split data arrays with columns instead of dataloaders. Defaults to True.
        seed (int, optional): random state. Defaults to 42.

    Returns:
        _type_: Dataloaders or data arrays with column names.
    """
    # Identify numeric columns (assuming non-numeric are object types or datetime)
    df = df_in.select_dtypes(include=["number"])
    # Scale only numeric columns
    scaler = StandardScaler()
    # df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    dfvals = df.values
    output = output_in.values
    df_scaled = scaler.fit_transform(dfvals)

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        df_scaled, output, test_size=0.2, random_state=seed
    )
    if return_dfs:
        X_train_df =pd.DataFrame(X_train, columns=df.columns)
        X_test_df =pd.DataFrame(X_test, columns=df.columns)
        y_train_df =pd.DataFrame(y_train, columns=output_in.columns)
        y_test_df =pd.DataFrame(y_test, columns=output_in.columns)
        print("Returning DFs")
        return X_train_df, X_test_df, y_train_df, y_test_df
    else:
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
        batch_size = 2000
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
