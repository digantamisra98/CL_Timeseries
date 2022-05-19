import pickle
import random
from time import time
from typing import Union

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.nn.functional import mse_loss, l1_loss, binary_cross_entropy, cross_entropy
from torch.optim import Optimizer


class NBeatsNet(nn.Module):
    SEASONALITY_BLOCK = "seasonality"
    TREND_BLOCK = "trend"
    GENERIC_BLOCK = "generic"

    def __init__(
        self,
        device=torch.device("cpu"),
        stack_types=(TREND_BLOCK, SEASONALITY_BLOCK),
        nb_blocks_per_stack=3,
        forecast_length=5,
        backcast_length=10,
        thetas_dim=(4, 8),
        share_weights_in_stack=False,
        hidden_layer_units=256,
        final_layer_units=512,
        nb_harmonics=None,
    ):
        super(NBeatsNet, self).__init__()
        self.forecast_length = forecast_length
        self.backcast_length = backcast_length
        self.hidden_layer_units = hidden_layer_units
        self.nb_blocks_per_stack = nb_blocks_per_stack
        self.share_weights_in_stack = share_weights_in_stack
        self.nb_harmonics = nb_harmonics
        self.stack_types = stack_types
        self.stacks = []
        self.thetas_dim = thetas_dim
        self.parameters = []
        self.final_layer_units = final_layer_units
        self.device = device
        print("| N-Beats")
        for stack_id in range(len(self.stack_types)):
            self.stacks.append(self.create_stack(stack_id))
        self.parameters = nn.ParameterList(self.parameters)
        self.final_layer = nn.Sequential(
            nn.Linear(self.hidden_layer_units, self.final_layer_units),
            nn.ReLU(),
            nn.Linear(self.final_layer_units, 2),
            nn.Softmax(dim=1),
        )

        self.to(self.device)
        self._loss = None
        self._opt = None

    def create_stack(self, stack_id):
        stack_type = self.stack_types[stack_id]
        print(
            f"| --  Stack {stack_type.title()} (#{stack_id}) (share_weights_in_stack={self.share_weights_in_stack})"
        )
        blocks = []
        for block_id in range(self.nb_blocks_per_stack):
            block_init = NBeatsNet.select_block(stack_type)
            if self.share_weights_in_stack and block_id != 0:
                block = blocks[-1]  # pick up the last one when we share weights.
            else:
                block = block_init(
                    self.hidden_layer_units,
                    self.thetas_dim[stack_id],
                    self.device,
                    self.backcast_length,
                    self.forecast_length,
                    self.nb_harmonics,
                )
                self.parameters.extend(block.parameters())
            print(f"     | -- {block}")
            blocks.append(block)
        return blocks

    def save(self, filename: str):
        torch.save(self, filename)

    @staticmethod
    def load(f, map_location=None, pickle_module=pickle, **pickle_load_args):
        return torch.load(f, map_location, pickle_module, **pickle_load_args)

    @staticmethod
    def select_block(block_type):
        if block_type == NBeatsNet.SEASONALITY_BLOCK:
            return SeasonalityBlock
        elif block_type == NBeatsNet.TREND_BLOCK:
            return TrendBlock
        else:
            return GenericBlock

    def forward(self, backcast):
        backcast = squeeze_last_dim(backcast)
        # forecast = torch.zeros(size=(backcast.size()[0], self.forecast_length,))  # maybe batch size here.
        for stack_id in range(len(self.stacks)):
            for block_id in range(len(self.stacks[stack_id])):
                # get the hidden unit feature representation of the block.
                block = self.stacks[stack_id][block_id]
                backcast = block(backcast)

        # apply the final linear layer for binary classification
        forecast = self.final_layer(backcast)
        return forecast

        # b, f = self.stacks[stack_id][block_id](backcast)
        # backcast = backcast.to(self.device) - b
        # forecast = forecast.to(self.device) + f
        # return backcast, forecast


def squeeze_last_dim(tensor):
    if len(tensor.shape) == 3 and tensor.shape[-1] == 1:  # (128, 10, 1) => (128, 10).
        return tensor[..., 0]
    return tensor


def seasonality_model(thetas, t, device):
    p = thetas.size()[-1]
    assert p <= thetas.shape[1], "thetas_dim is too big."
    p1, p2 = (p // 2, p // 2) if p % 2 == 0 else (p // 2, p // 2 + 1)
    s1 = torch.tensor([np.cos(2 * np.pi * i * t) for i in range(p1)]).float()  # H/2-1
    s2 = torch.tensor([np.sin(2 * np.pi * i * t) for i in range(p2)]).float()
    S = torch.cat([s1, s2])
    return thetas.mm(S.to(device))


def trend_model(thetas, t, device):
    p = thetas.size()[-1]
    assert p <= 4, "thetas_dim is too big."
    T = torch.tensor([t**i for i in range(p)]).float()
    return thetas.mm(T.to(device))


def linear_space(backcast_length, forecast_length, is_forecast=True):
    horizon = forecast_length if is_forecast else backcast_length
    return np.arange(0, horizon) / horizon


class Block(nn.Module):
    def __init__(
        self,
        units,
        thetas_dim,
        device,
        backcast_length=10,
        forecast_length=5,
        share_thetas=False,
        nb_harmonics=None,
    ):
        super(Block, self).__init__()
        self.units = units
        self.thetas_dim = thetas_dim
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.share_thetas = share_thetas
        self.fc1 = nn.Linear(backcast_length, units)
        self.fc2 = nn.Linear(units, units)
        self.fc3 = nn.Linear(units, units)
        self.fc4 = nn.Linear(units, units)
        self.device = device
        self.backcast_linspace = linear_space(
            backcast_length, forecast_length, is_forecast=False
        )
        self.forecast_linspace = linear_space(
            backcast_length, forecast_length, is_forecast=True
        )
        if share_thetas:
            self.theta_f_fc = self.theta_b_fc = nn.Linear(units, thetas_dim, bias=False)
        else:
            self.theta_b_fc = nn.Linear(units, thetas_dim, bias=False)
            self.theta_f_fc = nn.Linear(units, thetas_dim, bias=False)

    def forward(self, x):
        x = squeeze_last_dim(x)
        x = F.relu(self.fc1(x.to(self.device)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x

    def __str__(self):
        block_type = type(self).__name__
        return (
            f"{block_type}(units={self.units}, thetas_dim={self.thetas_dim}, "
            f"backcast_length={self.backcast_length}, forecast_length={self.forecast_length}, "
            f"share_thetas={self.share_thetas}) at @{id(self)}"
        )


class SeasonalityBlock(Block):
    def __init__(
        self,
        units,
        thetas_dim,
        device,
        backcast_length=10,
        forecast_length=5,
        nb_harmonics=None,
    ):
        if nb_harmonics:
            super(SeasonalityBlock, self).__init__(
                units,
                nb_harmonics,
                device,
                backcast_length,
                forecast_length,
                share_thetas=True,
            )
        else:
            super(SeasonalityBlock, self).__init__(
                units,
                forecast_length,
                device,
                backcast_length,
                forecast_length,
                share_thetas=True,
            )

    def forward(self, x):
        x = super(SeasonalityBlock, self).forward(x)
        backcast = seasonality_model(
            self.theta_b_fc(x), self.backcast_linspace, self.device
        )
        forecast = seasonality_model(
            self.theta_f_fc(x), self.forecast_linspace, self.device
        )
        return backcast, forecast


class TrendBlock(Block):
    def __init__(
        self,
        units,
        thetas_dim,
        device,
        backcast_length=10,
        forecast_length=5,
        nb_harmonics=None,
    ):
        super(TrendBlock, self).__init__(
            units,
            thetas_dim,
            device,
            backcast_length,
            forecast_length,
            share_thetas=True,
        )

    def forward(self, x):
        x = super(TrendBlock, self).forward(x)
        backcast = trend_model(self.theta_b_fc(x), self.backcast_linspace, self.device)
        forecast = trend_model(self.theta_f_fc(x), self.forecast_linspace, self.device)
        return backcast, forecast


class GenericBlock(Block):
    def __init__(
        self,
        units,
        thetas_dim,
        device,
        backcast_length=10,
        forecast_length=5,
        nb_harmonics=None,
    ):
        super(GenericBlock, self).__init__(
            units, thetas_dim, device, backcast_length, forecast_length
        )

        self.backcast_fc = nn.Linear(thetas_dim, backcast_length)
        self.forecast_fc = nn.Linear(thetas_dim, forecast_length)

    def forward(self, x):
        # no constraint for generic arch.
        x = super(GenericBlock, self).forward(x)

        # theta_b = self.theta_b_fc(x)
        # theta_f = self.theta_f_fc(x)

        # backcast = self.backcast_fc(theta_b)  # generic. 3.3.
        # forecast = self.forecast_fc(theta_f)  # generic. 3.3.

        # return backcast, forecast

        return x
