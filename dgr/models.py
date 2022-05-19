from functools import reduce
import torch
from torch import nn
import dgr
import utils


# class Generator   (define your generative model here):
class Generator(nn.Module):
    def __init__(self, sequence_length, hidden_sizes):
        """
        :param channels: a list containing all channels including the input image channel (1 for gray, 3 for RGB)
        """
        super(Generator, self).__init__()
        assert isinstance(hidden_sizes, list)
        self.encoder = nn.Sequential()
        self.encoder.add_module("input", nn.Linear(sequence_length, hidden_sizes[0]))
        for i in range(1, len(hidden_sizes)):
            #  Each layer will divide the size of feature map by 2
            self.encoder.add_module(
                "linear%d" % i,
                nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]),
            )
            self.encoder.add_module("relu%d" % i, nn.ReLU(True))

        self.decoder = nn.Sequential()
        hidden_sizes = list(reversed(hidden_sizes))
        for i in range(len(hidden_sizes) - 1):
            # Each layer will double the size of feature map
            self.decoder.add_module(
                "inv-linear%d" % (i + 1),
                nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]),
            )
            self.decoder.add_module("relud%d" % i, nn.ReLU(True))
        self.decoder.add_module("output", nn.Linear(hidden_sizes[-1], sequence_length))

    def forward(self, x):
        h = self.encoder(x)
        y = self.decoder(h)
        return y


# class Solver (define your continual learner here):
class Solver(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(Solver, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
