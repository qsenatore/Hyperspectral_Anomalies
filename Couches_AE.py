#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 09:59:43 2025

@author: senatorequentin
"""

import spectral as spec
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
import os
import metrique
from scipy.spatial.distance import mahalanobis

# Premier AE avec 5 couches dans l'encodeur et 5 dans le décodeur

class AE_1(torch.nn.Module):
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
    
# Deuxième AE avec 3 couches dans l'encodeur et 3 dans le décodeur
    
class AE_2(torch.nn.Module):
    def __init__(self, n_pixels):
        super().__init__()

        self.n_pixels = n_pixels
        
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(n_pixels, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 8),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(8, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, n_pixels),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
# Troisième AE avec 2 couches dans l'encodeur et 2 dans le décodeur
    
class AE_3(torch.nn.Module):
    def __init__(self, n_pixels):
        super().__init__()

        self.n_pixels = n_pixels
        
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(n_pixels, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 8),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(8, 64),
            torch.nn.Linear(64, n_pixels),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
# Fonction pour entraîner le modèle
    
def train_DL_model(model, n_epochs, loader, loss_function, optimizer):
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

# Fonction pour réduire le nombre de bandes spectrales

def binning_spectral(X, nb_band_final):
    Nbband = X.shape[1]
    Nbpix = X.shape[0]
    im_bin = np.zeros((Nbpix, nb_band_final))

    band_width = int(Nbband / nb_band_final)
    for i in range(nb_band_final - 1):
        im_bin[:, i] = np.sum(X[:, i * band_width:(i + 1) * band_width], axis=1)

    im_bin[:, -1] = np.sum(X[:, (i + 1) * band_width:], axis=1)
    return im_bin

# Fonction pour calculer la distance RX de chaque pixel

def RX_global(im_bin):
    n_pix = im_bin.shape[0]
    centre = np.mean(im_bin, axis=0)
    distance = np.zeros(n_pix)
    matcov = np.cov(np.transpose(im_bin))
    inv_matcov = np.linalg.inv(matcov)
    # mat_centre=np.tile(centre, ( Nbpix,1))
    for i in range(n_pix):
        point = im_bin[i, :]
        distance[i] = mahalanobis(point, centre, inv_matcov)
    return distance

#os.environ['SPECTRAL_DATA'] = 'C:/Users/Admin/Documents/insa/4A/PIR/data/image2'
#os.environ['SPECTRAL_DATA'] = 'C:/Users/emaga/GIT/PIR/DATA0_INSA/image2'
os.environ['SPECTRAL_DATA'] = '/home/senatorequentin/Projet_RI/data'

pathproj = "scene_lac_berge.hdr"
pathproj2 = "scene_lac_berge_VT.hdr"

img = spec.open_image(pathproj)
proj = img.load()
proj = proj.squeeze()

img2=spec.open_image(pathproj2)
gt = img2.load()
gt = gt.squeeze()

Nbcol = proj.shape[1]
Nblig = proj.shape[0]
Nbpix = Nbcol * Nblig
Nbband = proj.shape[2]

n_bin = 30

data_im = np.reshape(proj, [Nbpix, Nbband])
im_bin = binning_spectral(data_im, n_bin)
distance = RX_global(im_bin)

batch_size = 32
n_epochs = 5
lr = 1e-3

# Fonction pour automatiser le test des différents AE

def test_autoencoder(model_class, X_train_tensor, n_bin):
    
    model = model_class(n_bin)
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)
    
    dataset = TensorDataset(X_train_tensor, X_train_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model, losses = train_DL_model(model, n_epochs, loader, loss_function, optimizer)
    diff_AE = X_train_tensor.numpy() - model(X_train_tensor).detach().numpy()
    diff_AE = diff_AE.reshape(Nblig, Nbcol, n_bin)
    
    return np.mean((diff_AE**2), axis=2)

# Fonction de test pour comparer les résultats de nos trois AE + le RX

def test_couches():
    
    # Liste des modèles
    
    models = [AE_1, AE_2, AE_3]
    
    # Initialisation de la liste des différences
    
    list_diff_AE = [distance]
    
    # Normalisation des données
    
    Scaler = MinMaxScaler()
    X_train_scaled = Scaler.fit_transform(im_bin)
    X_train_tensor = torch.from_numpy(X_train_scaled).float()
    
    # On teste chaque modèle
    
    for model_class in models:
        diff_AE = test_autoencoder(model_class, X_train_tensor, n_bin)
        list_diff_AE.append(diff_AE)

    # On plot les courbes ROC
    
    metrique.plot_multiple_roc_curves(gt, list_diff_AE, ['Rx', 'modèle 1', 'modèle 2', 'modèle 3'])


test_couches()