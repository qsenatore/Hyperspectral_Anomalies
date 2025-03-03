#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 17:31:32 2025

@authors : Ema, Julie, Quentin
"""

import spectral as spec
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
import os

class AE(torch.nn.Module):
    def __init__(self, n_pixels):
        super().__init__()

        self.n_pixels = n_pixels
        
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(n_pixels, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 4)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(4, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, n_pixels),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
def train_DL_model(model, n_epochs, loader, loss_function):
    losses = []
    for epoch in range(n_epochs):
       # print("epoch # ", epoch)
        for (image, _) in loader:

            reconstructed = model(image)

            loss = loss_function(reconstructed, image)

            optimizer.zero_grad()
            
            loss.backward()
            
            optimizer.step()

            losses.append(loss)

    return model, losses

def binning_spectral(X, nb_band_final):
    Nbband = X.shape[1]
    Nbpix = X.shape[0]
    im_bin = np.zeros((Nbpix, nb_band_final))

    band_width = int(Nbband / nb_band_final)
    for i in range(nb_band_final - 1):
        im_bin[:, i] = np.sum(X[:, i * band_width:(i + 1) * band_width], axis=1)

    im_bin[:, -1] = np.sum(X[:, (i + 1) * band_width:], axis=1)
    return im_bin

# os.environ['SPECTRAL_DATA'] = 'C:/Users/Admin/Documents/insa/4A/PIR/Hyperspectral_Anomalies/DATA0_INSA/image1'
# os.environ['SPECTRAL_DATA'] = 'C:/Users/emaga/GIT/PIR/DATA0_INSA/image1'
os.environ['SPECTRAL_DATA'] = '/home/senatorequentin/INSA/Projet_RI/data'
pathproj = "scene_lac.hdr"

img = spec.open_image(pathproj)
proj = img.load()
proj = proj.squeeze()

Nbcol = proj.shape[1]
Nblig = proj.shape[0]
Nbpix = Nbcol * Nblig
Nbband = proj.shape[2]

n_bin = 30

data_im = np.reshape(proj, [Nbpix, Nbband])

im_bin = binning_spectral(data_im, n_bin)

batch_size = 32
#n_epochs = 15
lr = 1e-3

model = AE(n_bin)
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=lr,
                             weight_decay=1e-8)

for n_epochs in range(1,16):
    print(f"Training epoch {n_epochs}/15...") 

    shuffle = True
    Scaler = MinMaxScaler()
    X_train_scaled = Scaler.fit_transform(im_bin)
    X_train_tensor = torch.from_numpy(X_train_scaled).float()
    dataset = TensorDataset(X_train_tensor, X_train_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    model, losses = train_DL_model(model, 1, loader, loss_function)
    
    diff_AE = X_train_scaled - model(X_train_tensor).detach().numpy()
    diff_AE = diff_AE.reshape(Nblig, Nbcol, n_bin)  
    diff_AE = np.mean((diff_AE**2),axis=2)
    
    fig, ax = plt.subplots()
    ax.imshow(diff_AE)
    plt.show()