import csv
import itertools
import os

import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import torch
from libs.DataPreparation import DataPreparation, Dataset
import matplotlib.pyplot as plt
from libs.Network import MultiLayersModel, custom_reset_parameters


def train(symbol_list, past_nums, prict_nums, num_epochs=10, lr=0.0005, batch_size=2, half_inner_layers=2,
          units_ratio=8, Draw=True):

    print(f"num_epochs={num_epochs}, lr={lr}, batch_size={batch_size}, half_inner_layers={half_inner_layers}, units_ratio={units_ratio}")
    data_preparation = DataPreparation()
    X, Y, means, stds, colums = data_preparation.processCsv(symbol_list, past_nums, prict_nums)
    Y = Y[:, :, 6]
    X_train, X_test = torch.split(X, int(X.shape[0] * 0.9), dim=0)
    Y_train, Y_test = torch.split(Y, int(Y.shape[0] * 0.9), dim=0)
    in_features = X_train.shape[1] * X_train.shape[2]
    out_features = Y_train.shape[1]
    model = MultiLayersModel(in_features, half_inner_layers, units_ratio, out_features)
    model.apply(custom_reset_parameters)

    loss_fn = nn.MSELoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.8, weight_decay=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)

    dataset = Dataset(X_train, Y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    min_loss = 1000000000.0
    model_save_path = "best_model.pth"
    #os.remove(model_save_path)
    cnt = 1
    loss_sum = 0
    loss_train = []

    for epoch in range(num_epochs):
        loss_sum = 0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # Forward pass
            inputs = inputs.flatten(start_dim=1, end_dim=-1)
            targets = targets.flatten(start_dim=1, end_dim=-1)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss_sum += loss.item()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

        loss_sum /= len(dataloader)

        optimizer.step()

        if min_loss > loss_sum:
            min_loss = loss_sum
            cnt = 1
            torch.save(model, model_save_path)
        else:
            cnt += 1


        scheduler.step()

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], "
            f"Loss: {loss_sum:.4f}, "
            f"Min_Loss: {min_loss:.4f}")

        if cnt > 0 and cnt % 10 == 0:
            model = torch.load(model_save_path)
            model.train()
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.8, weight_decay=0.001)
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
            print("loading model")

        if cnt == 30:
            print("reset lr and params")
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.8, weight_decay=0.001)
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
            model.apply(custom_reset_parameters)
            cnt = -50
            #scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
            #break

        if loss_sum > 1.5:
            loss_train.append(1.5)
        else:
            loss_train.append(loss_sum)

    model = torch.load(model_save_path)
    final_test_loss = evaluate_model(X_test, Y_test, model_save_path, Draw=Draw)
    print(f"final_evaluation_loss: {final_test_loss}")
    return model, loss_train, final_test_loss


def evaluate_model(X_test, Y_test, model_save_path, Draw = False):
    loss_fn = nn.MSELoss()
    model = torch.load(model_save_path)
    model.eval()

    Y_pred = model(X_test.flatten(start_dim=1, end_dim=-1))
    Y_pred = Y_pred.view(Y_test.shape[0], Y_test.shape[1])
    eval_loss = loss_fn(Y_pred, Y_test)

    if Draw:
        Y_test = Y_test.detach().numpy()
        Y_pred = Y_pred.detach().numpy()
        plt.figure(0)
        plt.plot(Y_test[0], color="green", linestyle="--", label="Ground Truth")
        plt.plot(Y_pred[0], color='red', linestyle='--', label='Prediction')
        plt.title(f"validation loss: {eval_loss.item()}")
        plt.xlabel("days")
        plt.ylabel("value")
        plt.legend()
        plt.show()


    return eval_loss.item()

def trainModel(symbol_list, past_nums, prict_nums, Draw = False):
    l_tr = None
    l_te = 1
    f_l = 10

    model_save_path = "final_model.pth"
    params = None
    num_epochs = [500]
    lrs = [0.01]
    batch_sizes = [128]
    half_layers = [2]
    units_ratios = [8]
    combinations = list(itertools.product(num_epochs, lrs, batch_sizes, half_layers, units_ratios))
    for num_epochs, lr, batch_size, half_inner_layers, units_ratio in combinations:

        model, loss_train, final_test_loss = train(symbol_list, past_nums, prict_nums, num_epochs=num_epochs,
                                                              lr=lr, batch_size=batch_size,
                                                              half_inner_layers=half_inner_layers,
                                                              units_ratio=units_ratio, Draw=True)
        if final_test_loss < f_l:
            f_l = final_test_loss
            l_tr = loss_train
            l_te = final_test_loss
            torch.save(model, model_save_path)
            params = [num_epochs, lr, batch_size, half_inner_layers, units_ratio]
        if f_l < 0.01:
            break

    """model, loss_train, final_test_loss = train(symbol_list, past_nums, prict_nums, num_epochs=params[0],
                                               lr=params[1], batch_size=params[2],
                                               half_inner_layers=params[3],
                                               units_ratio=params[4], Draw=True)"""
    print(f"the best set of params: {params}, final_test_loss: {final_test_loss}")
    if Draw:
        plt.figure(0)
        plt.plot(l_tr, color="green", linestyle="--", label="train_loss")
        plt.axhline(l_te, color='red', linestyle='-', label=f'final_test_loss: y={l_te}')
        plt.title("loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.show()




# Initialize model
if __name__ == '__main__':
    symbol_list = ["sh000001", "sh603259"]
    past_nums = 80
    prict_nums = 5

    trainModel(symbol_list, past_nums, prict_nums, Draw=True)
