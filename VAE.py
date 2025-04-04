#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 15:46:52 2025

@author: senatorequentin
"""

# Jusqu'à la classe VAE, tout ce code a été récupéré sur github

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
import os
import metrique
from scipy.spatial.distance import mahalanobis
import spectral as spec
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_curve, auc

class Encoder(torch.nn.Module):
    def __init__(self, D_in, H, D_out, keep_prob=0):
        super(Encoder, self).__init__()

        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, 32)

        self._enc_mu = torch.nn.Linear(32, D_out)
        self._enc_log_sigma = torch.nn.Linear(32, D_out)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))

        # print(x.shape)

        return self._enc_mu(x), self._enc_log_sigma(x)

class Decoder(torch.nn.Module):
    def __init__(self, D_in, H, D_out, keep_prob=0):
        super(Decoder, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        return F.relu(self.linear2(x))
    
def get_ae(encoder, decoder, x):
    # encoding
    mu, log_sigma = encoder(x)
    sigma = torch.exp(log_sigma)
    # sampling by re-parameterization technique
    z = mu + sigma * torch.randn_like(mu)

    # decoding
    y = decoder(z)
    # y = torch.clamp(y, 1e-8, 1 - 1e-8)

    return y

def get_z(encoder, x):

    # encoding
    mu, log_sigma = encoder(x)
    sigma = torch.exp(log_sigma)
    # sampling by re-parameterization technique
    z = mu + sigma * torch.randn_like(mu)

    return z

def get_loss(encoder, decoder, x, x_target):

    batchsz = x.size(0)
    # encoding
    mu, log_sigma = encoder(x)
    sigma = torch.exp(log_sigma)
    # sampling by re-parameterization technique
    z = mu + sigma * torch.randn_like(mu)

    # decoding
    y = decoder(z)
    # y = torch.clamp(y, 1e-8, 1 - 1e-8)


    # loss
    # marginal_likelihood2 = torch.sum(x_target * torch.log(y) + (1 - x_target) * torch.log(1 - y)) / batchsz
    # marginal_likelihood = -F.binary_cross_entropy(y, x_target, reduction='sum') / batchsz
    marginal_likelihood = -torch.pow(x_target - y, 2).sum() / batchsz
    # print(marginal_likelihood2.item(), marginal_likelihood.item())

    KL_divergence = 0.5 * torch.sum(
                                torch.pow(mu, 2) +
                                torch.pow(sigma, 2) -
                                torch.log(1e-8 + torch.pow(sigma, 2)) - 1
                               ).sum() / batchsz

    ELBO = marginal_likelihood - KL_divergence

    loss = -ELBO

    return y, z, loss, marginal_likelihood, KL_divergence

# On crée une classe VAE en se servant de tout ce qui précède

class VAE(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def forward(self, x):
        mu, log_sigma = self.encoder(x)
        sigma = torch.exp(0.5 * log_sigma)  # 0.5 pour éviter une variance trop grande
        eps = torch.randn_like(sigma)  # Échantillonnage aléatoire
        z = mu + sigma * eps  # Reparamétrisation

        reconstructed = self.decoder(z)
        return reconstructed, mu, log_sigma
    
# On définit une nouvelle fonction de perte adaptée aux VAE
    
def vae_loss(x, reconstructed, mu, log_sigma):
    
    # Erreur de reconstruction
    reconstruction_loss = F.mse_loss(reconstructed, x, reduction='sum')

    # KL divergence
    kl_divergence = -0.5 * torch.sum(1 + log_sigma - mu.pow(2) - log_sigma.exp())
    
    beta = 5

    return reconstruction_loss + beta*kl_divergence

# On définit une nouvelle fonction d'entraînement

def train_vae(model, dataloader, n_epochs, optimizer, device):
    model.train()
    model.to(device)
    losses = []

    for epoch in range(n_epochs):
        epoch_loss = 0
        for x, _ in dataloader:
            x = x.to(device)  # Envoi des données sur GPU si dispo
            
            reconstructed, mu, log_sigma = model(x)

            loss = vae_loss(x, reconstructed, mu, log_sigma)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        losses.append(epoch_loss / len(dataloader))
    #print(f"Epoch {epoch+1}, Loss: {epoch_loss / len(dataloader):.4f}")

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
n_epochs = 1
lr = 1e-3
input_dim = n_bin
hidden_dim = 64
latent_dim = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Préparer les données

Scaler = MinMaxScaler()
X_train_scaled = Scaler.fit_transform(im_bin)
X_train_tensor = torch.from_numpy(X_train_scaled).float()

transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# Initialisation du modèle
vae = VAE(input_dim, hidden_dim, latent_dim).to(device)
optimizer = optim.Adam(vae.parameters(), lr=1e-3)

# Entraînement
trained_vae, losses = train_vae(vae, train_loader, n_epochs, optimizer, device)

reconstructed, _, _ = trained_vae(X_train_tensor)  # On ne garde que la reconstruction
diff_AE = X_train_scaled - reconstructed.detach().numpy()
diff_AE = diff_AE.reshape(Nblig, Nbcol, n_bin)
diff_AE = np.mean((diff_AE**2),axis=2)

metrique.plot_roc_curve(gt,distance)
metrique.plot_roc_curve(gt,diff_AE)

with torch.no_grad():
    mu, log_sigma = vae.encoder(X_train_tensor.to(device))
    sigma = torch.exp(log_sigma) 

plt.scatter(mu[:,0],mu[:,1],cmap='coolwarm', alpha=0.5)
plt.title("Projection de l'espace latent")

