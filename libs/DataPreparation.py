import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset


class DataPreparation:

    def processCsv(self, symbol_list, past_num, prict_num):
        # Simulate multi-feature time series data (e.g., 3 features)
        full_data = pd.read_csv(f"..\data\{symbol_list[0]}.csv")
        for symbol in symbol_list[1:]:
            data = pd.read_csv(f"..\data\{symbol}.csv")
            full_data = pd.merge(full_data, data, how="inner", on="date")

        data = full_data.drop(columns=['date', 'open_x', 'low_x', 'amount', 'outstanding_share'])
        colums = data.columns
        data = data.values
        #data = data[-500:]

        # the weights are set manually, onece the colums got changed, this should be changed, either
        #weights = np.array([0.5, 2.0, 0.5, 2.0, 2.0])
        stds = np.nanstd(data, axis=0)
        #stds = stds*weights
        means = np.nanmean(data, axis=0)

        data = (data -means)/stds



        # Create sequences
        X, Y = [], []
        for i in range(len(data) - past_num - prict_num):
            X.append(data[i: i + past_num, :])  # Shape: (input_window, num_features)
            Y.append(data[i + past_num: i + past_num + prict_num, :])  # Shape: (output_window, num_features)

        X = np.array(X)  # Shape: (num_samples, input_window, num_features)
        Y = np.array(Y)  # Shape: (num_samples, output_window, num_features)

        indices = np.random.permutation(X.shape[0])

        X = X[indices]
        Y = Y[indices]

        # Convert to PyTorch tensors
        X = torch.tensor(X, dtype=torch.float32)  # Shape: (num_samples, input_window, num_features)
        Y = torch.tensor(Y, dtype=torch.float32)  # Shape: (num_samples, output_window, num_features)


        return X, Y, means, stds, colums

class Dataset(Dataset):
    def __init__(self, inputs, targets):
        if len(inputs) != len(targets):
            raise Exception
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

if __name__ == '__main__':
    list = ["sh000001", "sh603259"]
    data_preparation = DataPreparation()
    data_preparation.processCsv(list, 30, 10)
