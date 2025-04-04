# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 11:18:13 2025

@author: julie
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import precision_recall_curve, f1_score


color=['red','grey','blue','green','yellow','purple','darkorange']
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
    


def plot_roc_curve(ground_truth, diff_AE):
    """
    Génère une courbe ROC en utilisant les erreurs de reconstruction
    pour identifier les anomalies par rapport à la vérité de terrain.
    
    :param ground_truth: Les valeurs réelles des anomalies (doivent être binaires)
    :param confusion_AE: L'erreur de reconstruction ou la mesure des anomalies
    :param threshold: Seuil pour classer une anomalie comme positive ou négative
    """
    
    # Aplatir les matrices pour les comparer
    ground_truth_flat = ground_truth.flatten()
    diff_AE_flat = diff_AE.flatten()

    # Créer des étiquettes binaires à partir de ground_truth (1 pour anomalie, 0 pour normal)
    # Ici, on suppose que ground_truth est déjà une matrice binaire (1 pour anomalie, 0 pour normal)
    y_true = ground_truth_flat

    # Calculer les scores (erreurs de reconstruction) pour chaque pixel
    # Par exemple, la reconstruction error peut être comparée à un seuil
    y_scores = diff_AE_flat  # ou tu peux utiliser l'erreur de reconstruction directement

    # Calculer la courbe ROC
    fpr, tpr, seuil = roc_curve(y_true, y_scores)

    # Calculer l'AUC (Area Under the Curve)
    roc_auc = auc(fpr, tpr)

    # Tracer la courbe ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
  
    plt.xscale('log')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

    print("AUC (Area Under Curve) : ", roc_auc)


def Precision_Recall(ground_truth, diff_AE):
     # Aplatir les matrices pour la comparaison
     gt_flat = ground_truth.flatten()
     diff_AE_flat = diff_AE.flatten()
     
     # Vérifier que les tailles correspondent
     if gt_flat.shape != diff_AE_flat.shape:
         print("Erreur : les dimensions ne correspondent pas !")
         return
     precision, recall, thresholds = precision_recall_curve(gt_flat, diff_AE_flat)
     
     # Calcul de l'AUC (aire sous la courbe Precision-Recall)
     auc_pr = auc(recall, precision)
     plt.figure(figsize=(8, 6))
     plt.plot(recall, precision, marker='.', label=f'AUC = {auc_pr:.2f}')
     plt.xlabel('Recall')
     plt.ylabel('Precision')
     plt.title('Precision-Recall Curve')
     plt.legend()
     plt.grid()
     plt.show()
     print("auc = ", auc_pr)
     
     #on priviligie un bon recall (détecter toutes les anomalies)

     cible_recall = 0.90  # Exemple : On veut capturer 90% des vraies anomalies
     idx = np.argmax(recall >= cible_recall)  # Cherche le plus petit seuil qui atteint ce rappel
     seuil_cible = thresholds[idx]

     print(f"Seuil optimal pour un Rappel de {cible_recall*100:.0f}% : {seuil_cible:.4f}")




def plot_multiple_roc_curves(ground_truth, list_diff_AE, labels):
    """
    Génère plusieurs courbes ROC sur un même graphe.
    
    :param ground_truth: Les valeurs réelles des anomalies (binaire 0 ou 1)
    :param list_diff_AE: Liste de matrices des erreurs de reconstruction
    :param labels: Liste de noms pour chaque courbe ROC
    """

    plt.figure(figsize=(8, 6))

    for i, diff_AE in enumerate(list_diff_AE):
        # Aplatir les matrices pour les comparer
        ground_truth_flat = ground_truth.flatten()
        diff_AE_flat = diff_AE.flatten()

        # Calculer la courbe ROC
        fpr, tpr, _ = roc_curve(ground_truth_flat, diff_AE_flat)
        roc_auc = auc(fpr, tpr)

        # Tracer la courbe ROC
        plt.plot(fpr, tpr, lw=2, label=f'{labels[i]} (AUC = {roc_auc:.2f})')

    plt.xscale('log')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc='lower right')
    plt.show()