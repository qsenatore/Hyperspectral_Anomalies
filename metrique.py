# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 11:18:13 2025

@author: Admin
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def Confusion(ground_truth, resultat_AE):
    # Aplatir les matrices pour la comparaison
    ground_truth_flat = ground_truth.flatten()
    resultat_AE_flat = resultat_AE.flatten()

    # Vérifier que les tailles correspondent
    if ground_truth_flat.shape != resultat_AE_flat.shape:
        print("Erreur : les dimensions ne correspondent pas !")
        return

    # Calculer la matrice de confusion
    cm = confusion_matrix(ground_truth_flat, resultat_AE_flat)

    # Affichage avec Matplotlib
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues, ax=ax)

    plt.title("Matrice de Confusion")
    plt.show()

    print("Matrice de confusion :\n", cm)

    
    