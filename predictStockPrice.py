import torch
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from libs.Network import LSTMModel

device = "cpu"

# Load data
data = pd.read_csv("YaoMinKangDe_19900101_21000118.csv")
prices = data['close'].values

# Scale data
scaler = MinMaxScaler()
scaled_prices = scaler.fit_transform(prices.reshape(-1, 1))

def create_sequences(data, seq_length):
    sequences, targets = [], []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        target = data[i+seq_length]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

seq_length = 30  # Use 30 days of data to predict the next price
X, y = create_sequences(scaled_prices, seq_length)

# Split data
train_size = int(len(X) * 0.8)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


# Define the model architecture (this should match the architecture used when training)
model = LSTMModel(input_dim=1, hidden_dim=64, output_dim=1).to(device)  # Use your actual model class and input/output sizes

# Load the state_dict
model.load_state_dict(torch.load('model.pth'))

# Set the model to evaluation mode (important for inference)
model.eval()
with torch.no_grad():
    y_pred = model(X_test.to(device)).cpu().numpy()
    y_test = y_test.numpy()

# Inverse scale predictions
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate RMSE
rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
print(f"RMSE: {rmse}")

# Ensure y_pred and y_test are numpy arrays and properly scaled (if necessary)
y_pred = y_pred.flatten()  # Flatten to 1D array if needed
y_test = y_test.flatten()  # Flatten to 1D array if needed

# Plot y_test (ground truth) vs. y_pred (predictions)
plt.figure(figsize=(12, 6))
plt.plot(y_test, label="Actual Prices", color="blue")
plt.plot(y_pred, label="Predicted Prices", color="red", linestyle="--")

# Add title and labels
plt.title("Actual vs Predicted Prices")
plt.xlabel("Time Steps")
plt.ylabel("Price")

# Add legend
plt.legend()
# Display the plot
plt.show()
