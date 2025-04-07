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
            torch.nn.Linear(64, 2),
            #torch.nn.ReLU(),
            #torch.nn.Linear(32, 16),
            #torch.nn.ReLU(),
            #torch.nn.Linear(16, 8),
            #torch.nn.ReLU(),
            #torch.nn.Linear(8, 4)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(2, 64),
            torch.nn.ReLU(),
            #torch.nn.Linear(8, 16),
            #torch.nn.ReLU(),
            #torch.nn.Linear(16, 32),
            #torch.nn.ReLU(),
            #torch.nn.Linear(32, 64),
            #torch.nn.ReLU(),
            torch.nn.Linear(64, n_pixels),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded  # Retourner les deux
    

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
    #centre = np.mean(im_bin, axis=0)
    centre = np.mean(im_bin,axis=0)
    distance = np.zeros(n_pix)
    matcov = np.cov(np.transpose(im_bin))
    inv_matcov = np.linalg.inv(matcov)
    # mat_centre=np.tile(centre, ( Nbpix,1))
    for i in range(n_pix):
        point = im_bin[i, :]
        distance[i] = mahalanobis(point, centre, inv_matcov)
    return distance



#os.environ['SPECTRAL_DATA'] = 'C:/Users/Admin/Documents/insa/4A/PIR/data/image2'
os.environ['SPECTRAL_DATA'] = 'C:/Users/emaga/GIT/PIR/DATA0_INSA/image2'
#os.environ['SPECTRAL_DATA'] = '/home/senatorequentin/Projet_RI/data'

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


    
#plt.scatter(list_AE) 
import matplotlib.pyplot as plt
import numpy as np

def test_latentRX(model_class, im_bin, n_bin, batch_size):
    
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(im_bin)  # Applique Min-Max Scaling
    
    # Conversion en tenseur PyTorch
    X_train_tensor = torch.from_numpy(X_train_scaled).float()  # Convertir en FloatTensor
    
    dataset = TensorDataset(X_train_tensor, X_train_tensor)  # (Entrée, Sortie identique car autoencodeur)
    loader = DataLoader(dataset, batch_size, shuffle=True)
    
    
    autoencoder = model_class(n_pixels=n_bin)  # Assure-toi que 20 correspond à tes features
    autoencoder.eval()
    
    latent_points = []
    
    for (image, _) in loader:
        encoded, _ = autoencoder(image)
        encoded = encoded.view(encoded.size(0), -1)  # [batch_size, 2]
        encoded_np = encoded.detach().cpu().numpy()
        latent_points.append(encoded_np)
    
    # Concatenation de tous les batches
    latent_all = np.concatenate(latent_points, axis=0)  # shape: [N, 2]
    
    # Moyenne et écart-type par dimension
    mean = np.mean(latent_all, axis=0)
    std = np.std(latent_all, axis=0)

    return RX_global(latent_all)


distanceRx=test_latentRX(AE_1,im_bin, n_bin, 32)
distance_carte = np.reshape(distance, [Nblig, Nbcol])
fig2, ax2 = plt.subplots()
ax2.imshow(distance_carte)