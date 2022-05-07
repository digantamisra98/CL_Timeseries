import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from nbeats_pytorch.model import NBeatsNet

# arbitrary data in the form of a dataframe
data = pd.DataFrame(...)

def window_data(data, window_size):
    # define a list to store the dataframes
    windows = []
    indices = []
    # loop over the dataframe
    for i in range(len(data) - window_size):
        # append the dataframe to the list
        windows.append(data[i:i + window_size])
        indices.append(i)
        
    # return the list of dataframes and the indices
    return windows, indices

def main(data, window_size = 10, replay_size = 2, window_selector = 5, random = True, contiguous = False, full = False):

    # get the dataframes and the indices
    windows, indices = window_data(data, window_size)

    # select train data to be the window based on the window_selector
    x = windows[window_selector]

    # select replay windows randomly prior to the window_selector
    if random:
        replay_windows = np.random.choice(windows[:window_selector], replay_size)
    # select replay windows contiguously prior to the window_selector
    elif contiguous:
        replay_windows = windows[:window_selector][:replay_size]
    # select replay windows from the entire dataset
    elif full:
        replay_windows = windows[:window_selector]
    
    # split train/test.
    c = int(len() * 0.8)
    x_train, y_train = x[:c], y[:c]
    x_test, y_test = x[c:], y[c:]

    # concatenate train with replay_windows
    x_train = np.concatenate((x_train, replay_windows))
    y_train = np.concatenate((y_train, replay_windows))

    # normalization.
    norm_constant = np.max(x_train)
    x_train, y_train = x_train / norm_constant, y_train / norm_constant
    x_test, y_test = x_test / norm_constant, y_test / norm_constant

    # model
    net = NBeatsNet(
        stack_types=(NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK),
        forecast_length=forecast_length,
        backcast_length=backcast_length,
        hidden_layer_units=128,
    )
    optimiser = optim.Adam(lr=1e-4, params=net.parameters())

    grad_step = 0
    for epoch in range(1000):
        # train.
        net.train()
        train_loss = []
        for x_train_batch, y_train_batch in data_generator(x_train, y_train, batch_size):
            grad_step += 1
            optimiser.zero_grad()
            _, forecast = net(torch.tensor(x_train_batch, dtype=torch.float).to(net.device))
            loss = F.mse_loss(forecast, torch.tensor(y_train_batch, dtype=torch.float).to(net.device))
            train_loss.append(loss.item())
            loss.backward()
            optimiser.step()
        train_loss = np.mean(train_loss)

        # test.
        net.eval()
        _, forecast = net(torch.tensor(x_test, dtype=torch.float))
        test_loss = F.mse_loss(forecast, torch.tensor(y_test, dtype=torch.float)).item()



if __name__ == '__main__':
    main()
