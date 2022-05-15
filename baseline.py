import torch
import numpy as np  
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import argparse

# create argparser
parser = argparse.ArgumentParser()
parser.add_argument('--n_samples', type=int, default=100)
parser.add_argument('--n_features', type=int, default=2)
parser.add_argument('--n_hidden', type=int, default=128)
parser.add_argument('--n_epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=32)

args = parser.parse_args()

def create_dataset(n_samples, n_features):
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples)
    return X, y

X, y = create_dataset(args.n_samples, args.n_features)

class VAE(nn.Module):
    def __init__(self, timeseries_length, h_dim, z_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(timeseries_length, h_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(h_dim, z_dim*2)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, timeseries_length),
            nn.Sigmoid()
        )
    
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.var(torch.randn(*mu.size()))
        z = mu + std * esp
        return z
    
    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=1)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


model = VAE(args.n_features, args.n_hidden, args.n_features)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


for epoch in range(args.n_epochs):
    for i in range(0, len(X), args.batch_size):
        batch_x = X[i:i+args.batch_size]
        #batch_x = torch.var(batch_x)
        recon_x, mu, logvar = model(batch_x)
        loss = loss_function(recon_x, batch_x, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(loss.item())


def generate_samples(model, n_samples):
    z = torch.var(torch.randn(n_samples, args.n_features))
    return model.decoder(z)










        









        







        
