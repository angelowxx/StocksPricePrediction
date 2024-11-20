import numpy
import torch.nn as nn


class MultiLayersModel(nn.Module):
    def __init__(self, input_size, half_layer_nums, multiply, output_size):
        super(MultiLayersModel, self).__init__()
        self.fc = nn.ModuleList()
        sizes = numpy.array([input_size * multiply**i for i in range(half_layer_nums)])
        for i in range(half_layer_nums-1):
            in_size = sizes[i]
            out_size = sizes[i+1]
            fc_cur = nn.Linear(in_size, out_size)
            activation_cur = nn.ReLU()
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
