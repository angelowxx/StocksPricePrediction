import csv
import itertools
import os

import numpy as np
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import torch
from libs.DataPreparation import DataPreparation, Dataset
import matplotlib.pyplot as plt
from libs.Network import MultiLayersModel, custom_reset_parameters, custom_loss


def train(symbol_list, past_nums, prict_nums, num_epochs=10, lr=0.0005, batch_size=2, half_inner_layers=2,
          units_ratio=8, Draw=False):

    print(f"num_epochs={num_epochs}, lr={lr}, batch_size={batch_size}, half_inner_layers={half_inner_layers}, units_ratio={units_ratio}")
    data_preparation = DataPreparation()
    X, Y, means, stds, colums = data_preparation.processCsv(symbol_list, past_nums, prict_nums)
    Y = Y[:, :, 6]
    X_train, X_test = torch.split(X, int(X.shape[0] * 0.9), dim=0)
    Y_train, Y_test = torch.split(Y, int(Y.shape[0] * 0.9), dim=0)
    in_features = X_train.shape[1] * X_train.shape[2]
    out_features = Y_train.shape[1]
    model = MultiLayersModel(in_features, half_inner_layers, units_ratio, out_features)
    #model.apply(custom_reset_parameters)

    #loss_fn = nn.MSELoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.8, weight_decay=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)

    dataset = Dataset(X_train, Y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    min_loss = 1000000000.0
    pre_loss = min_loss
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
            loss = custom_loss(outputs, targets)
            loss_sum += loss.item()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_sum /= len(dataloader)
        scheduler.step()


        if min_loss > loss_sum:
            min_loss = loss_sum
            torch.save(model, model_save_path)

        if pre_loss > loss_sum:
            cnt = 0
        else:
            cnt += 1

        pre_loss = loss_sum




        print(
            f"Epoch [{epoch + 1}/{num_epochs}], "
            f"Loss: {loss_sum:.4f}, "
            f"Min_Loss: {min_loss:.4f}")

        if cnt > 5:
            print("early stop!!")
            break

        if loss_sum > 1.5:
            loss_train.append(1.5)
        else:
            loss_train.append(loss_sum)

        if loss_sum < 0.01:
            print("early stop!!")
            break

    model = torch.load(model_save_path, weights_only=False)
    final_test_loss = evaluate_model(X_test, Y_test, stds, means, model_save_path, Draw=Draw)

    print(f"final_evaluation_loss: {final_test_loss}")
    return model, loss_train, final_test_loss


def evaluate_model(X_test, Y_test, stds, means, model_save_path, Draw = False):
    loss_fn = nn.L1Loss()
    model = torch.load(model_save_path, weights_only=False)
    model.eval()

    Y_pred = model(X_test.flatten(start_dim=1, end_dim=-1))
    Y_pred = Y_pred.view(Y_test.shape[0], Y_test.shape[1])
    eval_loss = loss_fn(Y_pred, Y_test)

    Y_test = Y_test.detach().numpy()
    Y_pred = Y_pred.detach().numpy()
    Y_pred = Y_pred * stds[6] + means[6]
    Y_test = Y_test * stds[6] + means[6]
    pred = Y_pred[:, len(Y_pred[0])-1]
    test = Y_test[:, len(Y_pred[0])-1]
    errors = np.absolute(pred - test)
    accepted_rate_1 = np.sum(errors < 1) / len(errors)
    accepted_rate_2 = np.sum(errors < 2) / len(errors)
    accepted_rate_3 = np.sum(errors < 3) / len(errors)

    for i in range(0, 101, 10):
        pred = Y_pred[i, :]
        test = Y_test[i, :]
        errors = np.absolute(pred-test)
        mean_error = errors.mean()
        """accepted_rate_1 = np.sum(errors < 1)/len(errors)
        accepted_rate_2 = np.sum(errors < 2) / len(errors)
        accepted_rate_3 = np.sum(errors < 3) / len(errors)"""
        plt.figure(0)
        plt.plot(test, color="green", linestyle="--", label="Ground Truth")
        plt.plot(pred, color='red', linestyle='--', label='Prediction')
        plt.title(f"compaison of future {len(pred)} days, mean error: {mean_error:.4f}\n"
                  #f"ratio of the 30th day with absolute error less than 1 to total : {accepted_rate_1}\n"
                  f"ratio of the {len(pred)}th day with absolute error less than 2 to total : {accepted_rate_2}\n"
                  f"ratio of the {len(pred)}th day with absolute error less than 3 to total : {accepted_rate_3}")
        plt.xlabel("days")
        plt.ylabel("value")
        plt.legend()
        if Draw:
            plt.show()
        else:
            #plt.savefig(f"pred_day{i+1}.png")
            plt.close()


    return eval_loss.item()

def trainModel(symbol_list, past_nums, prict_nums, Draw = False):
    l_tr = None
    l_te = 1
    f_l = 10

    model_save_path = "final_model.pth"
    params = None
    num_epochs = [50]
    lrs = [0.0005]
    batch_sizes = [4]
    half_layers = [2]
    units_ratios = [4]
    combinations = list(itertools.product(num_epochs, lrs, batch_sizes, half_layers, units_ratios))
    for num_epochs, lr, batch_size, half_inner_layers, units_ratio in combinations:

        model, loss_train, final_test_loss = train(symbol_list, past_nums, prict_nums, num_epochs=num_epochs,
                                                              lr=lr, batch_size=batch_size,
                                                              half_inner_layers=half_inner_layers,
                                                              units_ratio=units_ratio, Draw=Draw)
        if final_test_loss < f_l:
            f_l = final_test_loss
            l_tr = loss_train
            l_te = final_test_loss
            torch.save(model, model_save_path)
            params = [num_epochs, lr, batch_size, half_inner_layers, units_ratio]
        if f_l < 0.01:
            break

        plt.figure(0)
        plt.plot(loss_train, color="green", linestyle="--", label="train_loss")
        plt.axhline(final_test_loss, color='red', linestyle='-', label=f'final_test_loss: y={l_te}')
        plt.title(f"loss_{num_epochs}_{lr}_{batch_size}_{half_inner_layers}_{units_ratio}")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        if Draw:
            plt.show()
        else:
            plt.savefig(f"../data/loss_{num_epochs}_{lr}_{batch_size}_{half_inner_layers}_{units_ratio}.png")
            plt.close()

    print(f"the best set of params: {params}, final_test_loss: {final_test_loss}")




# Initialize model
if __name__ == '__main__':
    symbol_list = ["sh000001", "sh603259"]
    past_nums = 80
    prict_nums = 10

    trainModel(symbol_list, past_nums, prict_nums, Draw=True)
