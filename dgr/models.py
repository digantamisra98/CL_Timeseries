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
    def __init__(self, sequence_length, hidden_sizes, classes):
        pass
