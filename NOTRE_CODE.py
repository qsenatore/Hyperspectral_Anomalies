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
from sklearn.metrics import confusion_matrix
from skimage.metrics import structural_similarity as ssim
import metrique


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

#os.environ['SPECTRAL_DATA'] = 'C:/Users/Admin/Documents/insa/4A/PIR/data/image2'
#os.environ['SPECTRAL_DATA'] = 'C:/Users/emaga/GIT/PIR/DATA0_INSA/image1'
os.environ['SPECTRAL_DATA'] = '/home/senatorequentin/INSA/Projet_RI/data'
pathproj = "scene_lac_berge.hdr"

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
n_epochs = 11
lr = 1e-3

model = AE(n_bin)

loss_function = torch.nn.MSELoss()
#loss_function = torch.nn.L1Loss()

optimizer = torch.optim.Adam(model.parameters(),
                             lr=lr,
                             weight_decay=1e-8)

for n_epochs in range(10,n_epochs):
    print(f"Nombre d'epochs : {n_epochs}") 

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
    
    #fig, ax = plt.subplots()
    #ax.imshow(diff_AE)
    #plt.show()
    
    
    
#métrique TABLE DE CONFUSION

print(np.max(diff_AE))


vecteur_seuil = np.linspace(0,np.max(diff_AE),100) #seuil a faire varier pour determiner la sensibiliter de la detection d'anomalie
taux_detection = []

pathproj2 = "scene_lac_berge_VT.hdr"
img2=spec.open_image(pathproj2)
gt = img2.load()
gt = gt.squeeze()

for seuil in vecteur_seuil:
    resultat_AE = np.where(diff_AE > seuil, 1, 0)  # 1 = anomalie, 0 = normal
    
#fig, ax = plt.subplots()
#affiche les anomalies detectées (seuil a faire varier)
#ax.imshow(resultat_AE, cmap='gray') #diff_AE mais binaire avec seuil
#plt.title("Carte des Anomalies")
#plt.show()

#affiche la matrice de confusion
#metrique.Confusion(gt, resultat_AE)
#metrique.plot_roc_curve(gt, diff_AE, seuil)

    ground_truth_flat = gt.flatten()
    resultat_AE_flat = resultat_AE.flatten()
    matrix = confusion_matrix(ground_truth_flat, resultat_AE_flat)
    
    taux_detection.append(np.trace(matrix)/np.sum(matrix))
    
taux_detection=np.array(taux_detection)

seuil = 0.99

plt.plot(vecteur_seuil,taux_detection, label="Courbe")

for i in range(1, len(vecteur_seuil)):
    if taux_detection[i] > seuil and taux_detection[i-1] <= seuil:
        plt.axvline(x=vecteur_seuil[i], color='red', linestyle='--', label='Taux > 0.99')
        plt.text(vecteur_seuil[i], taux_detection[i], f'{vecteur_seuil[i]}', color='green', verticalalignment='top', horizontalalignment='left')
        
plt.title("Précision en fonction du seuil")
plt.xlabel("Seuil")
plt.ylabel("Taux de bonnes détections")
plt.legend()





    