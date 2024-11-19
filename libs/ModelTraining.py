import torch.nn as nn
from libs.Network import TransformerMultiFeature
import torch
from libs.DataPreparation import DataPreparation
import matplotlib.pyplot as plt


symbol_list = ["sh000001", "sh603259"]
data_preparation = DataPreparation()
data, X, Y, stds, means = data_preparation.processCsv(symbol_list)

# Initialize model
num_features = data.shape[1]  # Number of features in the dataset
model = TransformerMultiFeature(num_features=num_features)

criterion = nn.MSELoss()  # Use MSE for regression tasks
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
batch_size = 32

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for i in range(0, len(X), batch_size):
        X_batch = X[i : i + batch_size]
        Y_batch = Y[i : i + batch_size]

        # Prepare target input and output for the decoder
        tgt = Y_batch[:, :-1, :]  # Remove the last step
        tgt_output = Y_batch[:, 1:, :]  # Remove the first step

        optimizer.zero_grad()
        output = model(X_batch, tgt)
        loss = criterion(output, tgt_output)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")


input_window = 50
output_window = 5
model.eval()
with torch.no_grad():
    # Provide an input sequence
    input_seq = torch.tensor(data[-input_window-output_window:len(data)-output_window, :], dtype=torch.float32).unsqueeze(0)  # Shape: (1, input_window, num_features)

    # Initialize target sequence with zeros
    tgt_seq = torch.zeros((1, output_window, num_features), dtype=torch.float32)  # Shape: (1, output_window-1, num_features)

    # Predict future sequence
    output = model(input_seq, tgt_seq).squeeze().numpy()

output = output * stds + means
data = data * stds + means

plt.plot(output[:, 2], color="red", linestyle="--")
plt.plot(data[-output_window:, 2], color="blue", linestyle="--")
# Add labels and title
plt.title("Sequence Plot")
plt.xlabel("Index")
plt.ylabel("Value")
plt.show()
print("Predicted sequence shape:", output.shape)

