import numpy as np
import torch
import pandas as pd

class DataPreparation:
    def processCsv(self, symbol_list):
        # Simulate multi-feature time series data (e.g., 3 features)
        full_data = pd.read_csv(f"..\data\{symbol_list[0]}.csv")
        for symbol in symbol_list[1:]:
            data = pd.read_csv(f"..\data\{symbol}.csv")
            full_data = pd.merge(full_data, data, how="inner", on="date")

        data = full_data.drop(columns=['date', 'open_x', 'open_y', 'high_x', 'high_y', 'low_x', 'low_y', 'amount']).values

        stds = np.nanstd(data, axis=0)
        means = np.nanmean(data, axis=0)

        data = (data -means)/stds


        input_window = 50  # Number of historical steps
        output_window = 5  # Number of future steps

        # Create sequences
        X, Y = [], []
        for i in range(len(data) - input_window - output_window):
            X.append(data[i: i + input_window, :])  # Shape: (input_window, num_features)
            Y.append(
                data[i + input_window: i + input_window + output_window, :])  # Shape: (output_window, num_features)

        X = np.array(X)  # Shape: (num_samples, input_window, num_features)
        Y = np.array(Y)  # Shape: (num_samples, output_window, num_features)

        # Convert to PyTorch tensors
        X = torch.tensor(X, dtype=torch.float32)  # Shape: (num_samples, input_window, num_features)
        Y = torch.tensor(Y, dtype=torch.float32)  # Shape: (num_samples, output_window, num_features)


        return data, X, Y, stds, means



if __name__ == '__main__':
    list = ["sh000001", "sh603259"]
    data_preparation = DataPreparation()
    data_preparation.processCsv(list)
