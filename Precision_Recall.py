#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 11:18:42 2025

@author: sdoz
"""

import spectral as spec
import numpy as np
from scipy.spatial.distance import mahalanobis
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
import os
from sklearn.metrics import precision_recall_curve, auc



class AE(torch.nn.Module):
    def __init__(self, n_pixels):
        super().__init__()

        self.n_pixels = n_pixels
        
        # Building a linear encoder with Linear
        # layer followed by Relu activation function
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


        # Building an linear decoder with Linear
        # layer followed by Relu activation function
        # The Sigmoid activation function
        # outputs the value between 0 and 1
        # 9 ==> 784
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
        print("epoch # ", epoch)
        for (image, _) in loader:
            # Output of Autoencoder
            reconstructed = model(image)
            # Calculating the loss function
            loss = loss_function(reconstructed, image)

            # The gradients are set to zero,
            # the gradient is computed and stored.
            # .step() performs parameter update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Storing the losses in a list for plotting
            losses.append(loss)

    return model, losses



def binning_spectral(X, nb_band_final):
    # l'étape par reduction de bande permet de reduire la compléxité calculatoire
    # et de se rapprocher de l'hypothèse d'independance des données
    Nbband = X.shape[1]
    Nbpix = X.shape[0]
    im_bin = np.zeros((Nbpix, nb_band_final))

    band_width = int(Nbband / nb_band_final)
    for i in range(nb_band_final - 1):
        im_bin[:, i] = np.sum(X[:, i * band_width:(i + 1) * band_width], axis=1)

    im_bin[:, -1] = np.sum(X[:, (i + 1) * band_width:-1], axis=1)
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


if __name__ == "__main__":
   # pathproj='/run/media/sdoz/One Touch/INSA/scene_lac.hdr'
    
   # os.environ['SPECTRAL_DATA'] = 'C:/Users/Admin/Documents/insa/4A/PIR/Hyperspectral_Anomalies/DATA0_INSA/image1'
    os.environ['SPECTRAL_DATA'] = 'C:/Users/emaga/GIT/PIR/DATA0_INSA/image2'
    #os.environ['SPECTRAL_DATA'] = '/home/senatorequentin/INSA/Projet_RI/data'
    pathproj = "scene_lac_berge.hdr"
    
    # LECTURE IMAGE
    img = spec.open_image(pathproj)
    proj = img.load()
    proj = proj.squeeze()

    #donne des infos caractéristiques de l'img (nb col, nb ligne, nb pixels, nb bandes)
    Nbcol = proj.shape[1]
    Nblig = proj.shape[0]
    Nbpix = Nbcol * Nblig
    Nbband = proj.shape[2]

    n_bin = 30

    #ils mettent le cube en matrice (chaquse lignes representent un pixel avec dans chacunes des col une de ses bandes)
    data_im = np.reshape(proj, [Nbpix, Nbband])
    
    #image avec bandes regroupées : 
    im_bin = binning_spectral(data_im, n_bin)


    # RX method
    #calcul d'un vecteur distance qui stock toutes les distances pour chaques pixels (1 d pour 1 pix)
    distance = RX_global(im_bin)
    #reshape en mode matrice avec le bon nombre de ligne et de col 
    #c'est ce qui va être affiché
    distance_carte = np.reshape(distance, [Nblig, Nbcol])

    #première image que l'on affiche, on l'a recentrée avec data_im.max()
    data_img_matrix = data_im.reshape(Nblig, Nbcol, Nbband) / data_im.max()
    data_img_bin_matrix = im_bin.reshape(Nblig, Nbcol, n_bin)

    # we can do some anomaly detection using IF algorithm
    #Return the anomaly score of each sample using the IsolationForest algorithm
    #The IsolationForest ‘isolates’ observations by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature.
    #Since recursive partitioning can be represented by a tree structure, the number of splittings required to isolate a sample is equivalent to the path length from the root node to the terminating node.
    #This path length, averaged over a forest of such random trees, is a measure of normality and our decision function.
    #Random partitioning produces noticeably shorter paths for anomalies. Hence, when a forest of random trees collectively produce shorter path lengths for particular samples, they are highly likely to be anomalies.
    
    #n-jos=-1 => utilise tout les coeurs de l'ordi pour une meilleur perf
    #n-estimators : nombre d'arbres que l'on fait
    #max_samples: chaque arbre est entrainé qu'à 70% sur les données (permet d'éviter le sur apprentissage)
   
    IF = IsolationForest(n_jobs=-1, n_estimators=100, max_samples=0.7)
    IF.fit(im_bin)
    IF_scores = IF.score_samples(im_bin)
    im_IF_score = IF_scores.reshape(Nblig, Nbcol)
    # im_IF_score = np.where(im_IF_score < -0.65, 1, 0)


    # DL training a AE

    # Some parameters to be changed
    batch_size = 32
    n_epochs = 15
    lr = 1e-3

    shuffle = True
    Scaler = MinMaxScaler()
    X_train_scaled = Scaler.fit_transform(im_bin)
    X_train_tensor = torch.from_numpy(X_train_scaled).float()
    dataset = TensorDataset(X_train_tensor, X_train_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    # Model Initialization
    model = AE(n_bin)
    # Validation using MSE Loss function // Can be changed !
    loss_function = torch.nn.MSELoss()
    # Using an Adam Optimizer with lr to be fixed
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 weight_decay=1e-8)

    model, losses = train_DL_model(model, n_epochs, loader, loss_function)
    
    #erreure de reconstruction
    diff_AE = X_train_scaled - model(X_train_tensor).detach().numpy()
    diff_AE = diff_AE.reshape(Nblig, Nbcol, n_bin)
    
    diff_AE = np.mean((diff_AE**2),axis=2)



    # plot some figure as exemple
    print("size img", data_im.shape)
    print("size img binned", im_bin.shape)
    print("distance_carte", distance_carte.shape)  
  

    fig, ax = plt.subplots()
    ax.imshow(data_img_matrix[:, :, [85, 50, 14]])

    fig2, ax2 = plt.subplots()
    ax2.imshow(distance_carte)

    fig3, ax3 = plt.subplots()
    ax3.imshow(im_IF_score)
    
    fig4, ax4 = plt.subplots()
    ax4.imshow(diff_AE)
    plt.show()