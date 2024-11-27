import numpy
import torch
import torch.nn as nn
import torch.nn.init as init


class MultiLayersModel(nn.Module):
    def __init__(self, input_size, half_layer_nums, multiply, output_size):
        super(MultiLayersModel, self).__init__()
        self.fc = nn.ModuleList()
        sizes = numpy.array([input_size * (int)(multiply**i) for i in range(half_layer_nums)])
        for i in range(half_layer_nums-1):
            in_size = sizes[i]
            out_size = sizes[i+1]
            fc_cur = nn.Linear(in_size, out_size)
            activation_cur = nn.ReLU()
            if i == 0:
                dropout = nn.Dropout(p=0.2)
            else:
                dropout = nn.Dropout(p=0.5)
            self.fc.append(fc_cur)
            self.fc.append(activation_cur)
            self.fc.append(dropout)

        for i in range(half_layer_nums-1, 0, -1):
            in_size = sizes[i]
            out_size = sizes[i - 1]
            fc_cur = nn.Linear(in_size, out_size)
            activation_cur = nn.ReLU()
            dropout = nn.Dropout(p=0.5)
            self.fc.append(fc_cur)
            self.fc.append(activation_cur)
            self.fc.append(dropout)
        fc_cur = nn.Linear(input_size, output_size)
        #activation_cur = nn.ReLU()
        self.fc.append(fc_cur)
        #self.fc.append(activation_cur)

    def forward(self, x):
        for fn in self.fc:
            x = fn(x)
        return x

def custom_reset_parameters(m):
    if isinstance(m, nn.Linear):  # Example for nn.Linear layers
        init.uniform_(m.weight, -0.01, 0.01)
        if m.bias is not None:
            init.uniform_(m.bias, -0.01, 0.01)

def custom_loss(y_pred, y_true, lambda_var=0.2):
    huber_loss = torch.nn.HuberLoss(delta=0.2)
    mse_loss = huber_loss(y_pred, y_true)
    variance = torch.var(y_pred, unbiased=False)  # Variance of predictions
    variance_true = torch.var(y_true, unbiased=False)
    variance -= variance_true
    variance = torch.abs(variance)
    return (1-lambda_var)*mse_loss + lambda_var * variance  # encourage prediction to have the same variance as y_true
