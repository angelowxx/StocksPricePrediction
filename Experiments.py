import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from libs.Network import LSTMModel
from libs.DataPreparation import DataPreparation
from libs.ModelTraining import ModelTraining
import joblib

def trainModel():
    # Load data
    path = "YaoMinKangDe_19900101_21000118.csv"
    target = "close"
    seq_len = 30
    dataPreparation = DataPreparation(path, target)
    train_loader, X_test, y_test, input_dim, output_dim = dataPreparation.process(seq_len)
    modelTraining = ModelTraining(train_loader, X_test, y_test, input_dim, output_dim)
    modelTraining.createModel(50)
    modelTraining.testModel()


def testModel(his_days, predict_days):
    # Reconstruct the model
    path = "YaoMinKangDe_19900101_21000118.csv"
    scaler = joblib.load('scaler.pkl')
    data = pd.read_csv(path)
    data = data.drop(columns=['date','volume','amount','outstanding_share']).values
    data = data[len(data) - his_days-1:]
    data = scaler.fit_transform(data)
    res = np.empty((0, 5))
    X = data[:len(data)-1]
    X = torch.tensor(np.array([X]), dtype=torch.float32)
    y = ModelTraining.predict(X, "cpu")
    y = scaler.inverse_transform(y)
    res = np.append(res, y, axis=0)
    data = data[len(data) - his_days:]

    for i in range(predict_days):
        X = data
        X = torch.tensor(np.array([X]), dtype=torch.float32)
        y = ModelTraining.predict(X, "cpu")
        y = scaler.inverse_transform(y)
        res = np.append(res, y, axis=0)
        data = np.append(data, y, axis=0)
        data = data[1:]

    plt.figure(figsize=(12, 6))
    plt.plot(res[:, 1], label="Predicted Prices", color="red")

    # Add title and labels
    plt.title(f"predicting tomorrow's price change : {res[1,1]-res[0,1]}")
    plt.xlabel("Time Steps")
    plt.ylabel("Price")

    # Add legend
    plt.legend()
    plt.tight_layout()

    # Display the plot
    plt.show()




if __name__ == '__main__':
    trainModel()
    testModel(2, 3)