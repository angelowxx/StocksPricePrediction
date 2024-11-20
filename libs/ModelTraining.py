import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from libs.Network import MultiLayersModel
import torch
from libs.DataPreparation import DataPreparation, Dataset
import matplotlib.pyplot as plt
from libs.Network import MultiLayersModel


def train(symbol_list, past_nums, prict_nums, num_epochs = 10):


    data_preparation = DataPreparation()
    X, Y, means, stds, colums = data_preparation.processCsv(symbol_list, past_nums, prict_nums)
    X_train, X_test = torch.split(X, int(X.shape[0] * 0.9), dim=0)
    Y_train, Y_test = torch.split(Y, int(Y.shape[0] * 0.9), dim=0)

    model = MultiLayersModel(X_train.shape[1]*X_train.shape[2], 2, 4, Y_train.shape[1]*Y_train.shape[2])

    loss_fn = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)

    dataset = Dataset(X_train, Y_train)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    for epoch in range(num_epochs):
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # Forward pass
            inputs = inputs.flatten(start_dim=1, end_dim=-1)
            targets = targets.flatten(start_dim=1, end_dim=-1)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    evaluate_model(model, X_test, Y_test, loss_fn, colums)
    return model

def evaluate_model(model, X_test, Y_test, loss_fn, colums):
    Y_pred = model(X_test.flatten(start_dim=1, end_dim=-1))
    Y_pred = Y_pred.view(Y_test.shape[0], Y_test.shape[1], Y_test.shape[2])
    eval_loss = loss_fn(Y_pred, Y_test)

    Y_pred = Y_pred.detach().numpy()
    Y_test = Y_test.detach().numpy()

    # close_market, volume_market, close_stock, volume_stock, outstanding_share_stock, turnover
    l = len(colums)
    width = 2
    height = (int)(l / width)
    if l % width != 0:
        height += 1

    fig, axis = plt.subplots(height, width, figsize=(10, 8))
    for i in range(height):
        for j in range(width):
            num = i * width + j
            if num >= l:
                break
            axis[i, j].plot(Y_test[0, :, num], color="red", linestyle="--", label="test data")
            axis[i, j].plot(Y_pred[0, :, num], color="blue", linestyle="--", label="pred data")
            axis[i, j].set_title(colums[num])
            axis[i, j].set_xlabel("predicted day")
            axis[i, j].set_ylabel("value")
            axis[i, j].legend()

    plt.tight_layout()
    fig.suptitle(f"The total evaluation loss: {eval_loss.item()}", fontsize=13)
    plt.show()



# Initialize model
if __name__ == '__main__':

    symbol_list = ["sh000001", "sh603259"]
    past_nums = 30
    prict_nums = 10



    train(symbol_list, past_nums, prict_nums, 100)
