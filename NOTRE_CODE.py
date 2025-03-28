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
import metrique
from scipy.spatial.distance import mahalanobis

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
    
def train_DL_model(model, n_epochs, loader, loss_function,optimizer):
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

def RX_global(im_bin):
    n_pix = im_bin.shape[0]
    centre = np.mean(im_bin, axis=0)
    distance = np.zeros(n_pix)
    matcov = np.cov(np.transpose(im_bin))
    inv_matcov = np.linalg.inv(matcov)
    # mat_centre=np.tile(centre, ( Nbpix,1))
    for i in range(n_pix):
        point = im_bin[i, :]
        #distance de malanobis pour chaque pixel on compare chacunes de ses longueurs d'ondes avec les longueurs d'ondes moyennes
        #on utilise l'inverse de la mat cov dans le calcul de malanobis
        distance[i] = mahalanobis(point, centre, inv_matcov)
    return distance

#Chargement des images :
    
os.environ['SPECTRAL_DATA'] = 'C:/Users/Admin/Documents/insa/4A/PIR/data/image2'
#os.environ['SPECTRAL_DATA'] = 'C:/Users/emaga/GIT/PIR/DATA0_INSA/image2'
#os.environ['SPECTRAL_DATA'] = '/home/senatorequentin/INSA/Projet_RI/data'
pathproj = "scene_lac_berge.hdr"
pathproj2 = "scene_lac_berge_VT.hdr"
img2=spec.open_image(pathproj2)
gt = img2.load() #Vérité Terrain
gt = gt.squeeze() 
img = spec.open_image(pathproj)
proj = img.load()
proj = proj.squeeze()

Nbcol = proj.shape[1]
Nblig = proj.shape[0]
Nbpix = Nbcol * Nblig
Nbband = proj.shape[2]

n_bin = 30

#on fait un Algo RX qui nous servira de base de comparaison

#ils mettent le cube en matrice (chaquse lignes representent un pixel avec dans chacunes des col une de ses bandes)
data_im = np.reshape(proj, [Nbpix, Nbband])

#image avec bandes regroupées : 
im_bin = binning_spectral(data_im, n_bin)

#calcul d'un vecteur distance qui stock toutes les distances pour chaques pixels (1 d pour 1 pix)
distanceRX = RX_global(im_bin)


#maintenant on passe aux AE

batch_size = 32

lr = 1e-3

model = AE(n_bin)
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-8)


def test_epochs(nepochs,batch_size):
    model = AE(n_bin)
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=1e-8)
    list_diff_AE=[] #liste qui va stocker tous les résultats diff_AE que l'on va ensuite comparer
    list_diff_AE.append(distanceRX) #on ajoute le résultat par RXglobal
    
    shuffle = True
    Scaler = MinMaxScaler()
    X_train_scaled = Scaler.fit_transform(im_bin)
    X_train_tensor = torch.from_numpy(X_train_scaled).float()
    dataset = TensorDataset(X_train_tensor, X_train_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    for n_epochs in range(1,nepochs+1):
        print(f"Training  {n_epochs} epoch(s)...") 
        
        #Entrainemnt du modèle
        model, losses = train_DL_model(model, 1, loader, loss_function,optimizer)
        X_reconstructed=model(X_train_tensor).detach().numpy()
        
        #Calcul de l'erreur de reconstruction
        diff_AE = X_train_scaled - X_reconstructed
        diff_AE = diff_AE.reshape(Nblig, Nbcol, n_bin)  
        diff_AE = np.mean((diff_AE**2),axis=2) #MSE 
        
        # Normalisation
        diff_AE = (diff_AE - np.min(diff_AE)) / (np.max(diff_AE) - np.min(diff_AE))
       
        # Stockage du résultat
        list_diff_AE.append(diff_AE.copy())
   
    # Affichage des courbes ROC
    metrique.plot_multiple_roc_curves(gt, list_diff_AE, ['Rx'] + [f'{i} epochs' for i in range(1, nepochs + 1)])

       


#test_epochs(3,32)



def test_batch(list_batch_size):
    n_epochs=1 
    list_diff_AE=[] #liste qui va stocker tous les résultats diff_AE que l'on va ensuite comparer
    list_diff_AE.append(distanceRX) #on ajoute le résultat par RXglobal  
    for batch_size in list_batch_size :
        print(f"Training  {n_epochs} epoch(s) with batch size {batch_size}...") 
        model = AE(n_bin)
        loss_function = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=1e-8)
       
        #paramètres pour la boucle de variation des batchs sizes
        shuffle = True
        Scaler = MinMaxScaler()
        X_train_scaled = Scaler.fit_transform(im_bin)
        X_train_tensor = torch.from_numpy(X_train_scaled).float()
        dataset = TensorDataset(X_train_tensor, X_train_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
        model, losses = train_DL_model(model, n_epochs, loader, loss_function,optimizer)
        
        diff_AE = X_train_scaled - model(X_train_tensor).detach().numpy()
        diff_AE = diff_AE.reshape(Nblig, Nbcol, n_bin)  
        diff_AE = np.mean((diff_AE**2),axis=2) #metrique à modifier avec un truc prenant en compte l'angle spectral
       
    
        list_diff_AE.append(diff_AE) #stockage  des résultats de chaque itération dans une liste
        
        #normaliser si valeurs trop proches
        diff_AE = (diff_AE - np.min(diff_AE)) / (np.max(diff_AE) - np.min(diff_AE))
        diff_AE_log = np.log1p(diff_AE)
        seuil = np.percentile(diff_AE_log, 99.9)
        resultat_AE = np.where(diff_AE_log > seuil, 1, 0) #pour faire afficher la carte des anomalies si besoin
        #metrique.plot_roc_curve(gt, diff_AE, seuil)
        
    metrique.plot_multiple_roc_curves(gt, list_diff_AE, ['Rx'] + [f'{b} batchs' for b in list_batch_size])

list_batch_size=[12,32,64,128]
test_batch(list_batch_size)


#On veut maintenant faire varier le calcul de la distance entre l'image original et l'image reconstruite

def spectral_angle_mapper(X_reconstructed, X_train_scaled):
    """
    Calcule l'angle spectral en radians entre deux vecteurs spectraux.
    """
    # Reshape pour correspondre à l'image originale
    X_original = X_train_scaled.reshape(Nblig, Nbcol, n_bin)
    X_reconstructed = X_reconstructed.reshape(Nblig, Nbcol, n_bin)

    # Produit scalaire entre les spectres originaux et reconstruits
    dot_product = np.sum(X_original * X_reconstructed, axis=2)

    # Normes des spectres originaux et reconstruits
    norm_orig = np.linalg.norm(X_original, axis=2)
    norm_recon = np.linalg.norm(X_reconstructed, axis=2)

    # Calcul du cosinus de l'angle spectral
    cos_theta = np.clip(dot_product / (norm_orig * norm_recon), -1, 1)

    # Calcul de l'angle spectral (SAM) en radians
    diff_AE_SAM = np.arccos(cos_theta)

    return diff_AE_SAM

def mahalanobis_distance(X_reconstructed, X_train_scaled):
    """
    Calcule la distance de Mahalanobis entre les pixels originaux et reconstruits.
    """
    # Reshape pour correspondre à l'image originale
    X_original = X_train_scaled.reshape(Nblig, Nbcol, n_bin)
    X_reconstructed = X_reconstructed.reshape(Nblig, Nbcol, n_bin)

    # Calcul de la matrice de covariance et de son inverse
    cov_matrix = np.cov(X_original.reshape(-1, n_bin).T)  # Covariance des bandes spectrales
    inv_cov_matrix = np.linalg.inv(cov_matrix + np.eye(n_bin) * 1e-6)  # Régularisation

    # Calcul pixel par pixel de la distance de Mahalanobis
    diff_AE_maha = np.zeros((Nblig, Nbcol))

    for i in range(Nblig):
        for j in range(Nbcol):
            diff_AE_maha[i, j] = mahalanobis(X_original[i, j], X_reconstructed[i, j], inv_cov_matrix)

    return diff_AE_maha

def test_distance():
    
    model = AE(n_bin)  # Définition du modèle
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-8)
    
    list_diff_AE = []  # Liste des résultats de distance
    list_diff_AE.append(distanceRX)  # Ajout du résultat RX global avec distance de maha
    
    # Paramètres d'entraînement
    n_epochs = 1
    batch_size = 32
    
    print(f"Training {n_epochs} epoch(s) with {batch_size} batch size...")

    # Prétraitement des données
    Scaler = MinMaxScaler()
    X_train_scaled = Scaler.fit_transform(im_bin)
    X_train_tensor = torch.from_numpy(X_train_scaled).float()
    
    # Chargement des données dans un DataLoader PyTorch
    dataset = TensorDataset(X_train_tensor, X_train_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Entraînement du modèle
    model, losses = train_DL_model(model, n_epochs, loader, loss_function)

    # Reconstruction avec l'AE
    X_reconstructed = model(X_train_tensor).detach().numpy()

    # MSE (Mean Squared Error)
    diff_AE_MSE = X_train_scaled - X_reconstructed
    diff_AE_MSE = diff_AE_MSE.reshape(Nblig, Nbcol, n_bin)  
    diff_AE_MSE = np.mean(diff_AE_MSE**2, axis=2)  # Moyenne sur les bandes spectrales
    list_diff_AE.append(diff_AE_MSE)

    #Spectral Angle Mapper (SAM)
    list_diff_AE.append(spectral_angle_mapper(X_reconstructed, X_train_scaled))

    #Distance de Mahalanobis
    list_diff_AE.append(mahalanobis_distance(X_reconstructed, X_train_scaled))

    # Affichage des courbes ROC
    metrique.plot_multiple_roc_curves(gt, list_diff_AE, ['Rx', 'MSE', 'SAM', 'Mahalanobis'])

#test_distance()


#print(np.max(diff_AE))


#seuil=0.009 #seuil a faire varier pour determiner la sensibiliter de la detection d'anomalie
#resultat_AE = np.where(diff_AE > seuil, 1, 0)  # 1 = anomalie, 0 = normal
#seuil=np.log1p(0.0698)



#fig, ax = plt.subplots()

#affiche les anomalies detectées (seuil a faire varier)
#ax.imshow(resultat_AE, cmap='gray') #diff_AE mais binaire avec seuil
#plt.title("Carte des Anomalies")
#plt.show()


#affiche la matrice de confusion
#metrique.Confusion(gt, resultat_AE)


#voir si les valeurs sont trop proche (et donc les seuils pris pour le PR ne sont pas pertinents)
#plt.hist(diff_AE.ravel(), bins=50)



#metrique.Precision_Recall(gt,diff_AE)






    