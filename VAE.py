#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 15:46:52 2025

@author: senatorequentin
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import metrique
from scipy.spatial.distance import mahalanobis
import spectral as spec
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

# Classe qui implémente l'Auto-Encodeur Variationnel (VAE)

class VAE(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, latent_dim):
        
        super(VAE, self).__init__()
        
        # Encodeur
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Décodeur
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
# Fonction de perte du VAE

def vae_loss(recon_x, x, mu, logvar):
    
    mse_loss = torch.nn.MSELoss()
    MSE = mse_loss(recon_x, x)    
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return MSE + 5*KLD

# Fonction d'entraînement du modèle

def train_DL_model(model, n_epochs, loader, loss_function, optimizer):
    
    losses = []
    
    for epoch in range(n_epochs):
        for (image, _) in loader:

            reconstructed = model(image)

            reconstructed, mu, logvar = model(image)
            loss = loss_function(reconstructed, image, mu, logvar)

            optimizer.zero_grad()
            
            loss.backward()
            
            optimizer.step()

            losses.append(loss)

    return model, losses

# Fonction qui permet de réduire le nombre de bandes spectrales utilisées

def binning_spectral(X, nb_band_final):
    
    Nbband = X.shape[1]
    Nbpix = X.shape[0]
    im_bin = np.zeros((Nbpix, nb_band_final))
    band_width = int(Nbband / nb_band_final)
    
    for i in range(nb_band_final - 1):
        
        im_bin[:, i] = np.sum(X[:, i * band_width:(i + 1) * band_width], axis=1)

    im_bin[:, -1] = np.sum(X[:, (i + 1) * band_width:], axis=1)
    
    return im_bin

# Fonction qui effectue l'algo RX pour une image

def RX_global(im_bin):
    
    n_pix = im_bin.shape[0]
    centre = np.mean(im_bin, axis=0)
    distance = np.zeros(n_pix)
    matcov = np.cov(np.transpose(im_bin))
    inv_matcov = np.linalg.inv(matcov)

    for i in range(n_pix):
        
        point = im_bin[i, :]
        distance[i] = mahalanobis(point, centre, inv_matcov)
        
    return distance

# Chargement des images
    
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

# Algo RX sur notre image

data_im = np.reshape(proj, [Nbpix, Nbband])
im_bin = binning_spectral(data_im, n_bin)
distanceRX = RX_global(im_bin)

# Paramètres du VAE

batch_size = 1000
lr = 1e-3
n_epochs = 1

# Initialisation

model = VAE(input_dim=n_bin, hidden_dim=16, latent_dim=2)
loss_function = vae_loss
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-8)

# Entraînement

Scaler = MinMaxScaler()
X_train_scaled = Scaler.fit_transform(im_bin)
X_train_tensor = torch.from_numpy(X_train_scaled).float()

dataset = TensorDataset(X_train_tensor, X_train_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model, losses = train_DL_model(model, n_epochs, loader, loss_function, optimizer)

reconstructed, _, _ = model(X_train_tensor)

# Récupération de l'espace latent

with torch.no_grad():
    X_encoded_mu, _ = model.encode(X_train_tensor)
    latent = X_encoded_mu.numpy()
    
# Moyenne et covariance de l'espace latent

center = np.mean(latent, axis=0)
cov = np.cov(latent.T)
inv_cov = np.linalg.inv(cov)

# Calcul de la distance de Mahalanobis pour chaque point

latent_distances = np.array([mahalanobis(z, center, inv_cov) for z in latent])
diff_VAE = latent_distances.reshape(Nblig, Nbcol) 

metrique.plot_roc_curve(gt, diff_VAE)

plt.scatter(latent[:, 0], latent[:, 1], s=2, alpha=0.5)
plt.xlabel("Latent dimension 1")
plt.ylabel("Latent dimension 2")
plt.title("Projection dans l’espace latent")
plt.grid(True)
plt.show()

plt.imshow(diff_VAE, cmap="inferno")
plt.colorbar(label="Distance de Mahalanobis (espace latent)")
plt.title("Carte d'anomalies via l'espace latent du VAE")
plt.axis("off")
plt.show()

diff_RX = distanceRX.reshape(Nblig, Nbcol)

plt.imshow(gt, cmap='inferno')
plt.title("Algo RX")
plt.axis("off")
plt.show()