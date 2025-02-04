#Tests librairie SPy

#Libraires nécessaires pour utiliser SPy

#classique, on a l'habitude
import numpy as np

#traitement d'images
from PIL import Image

#création d'interfaces grpahiques
# import wx as wx  (version trop récente de python pour cette bibliothèque -> essayer sur version 3.10)

#pour faire des graphiques
import matplotlib as plt

#environnement interactif nécessaire pour des interfaces graphiques fluides
import IPython as ip

#API pour du 2D ou 3D
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

#finalement, la libraire spectral
from spectral import *

#petite manip qui définit la variable d'environnement SPECTRAL_DATA qui définit un répertoire où trouver les images hyperspectrales
import os
os.environ['SPECTRAL_DATA'] = '/home/senatorequentin/INSA'

#ouvrir une image en python
img = open_image('92AV3C.lan')

print(img)

print()

#donne le type d'image (BIL,BIP,BSQ)
print("Type d'image : ")
print(img.__class__)
print()

#donne la forme de l'image (Lignes,Colonnes,Bandes)
print("Forme de l'image : ")
print(img.shape)
print()

#selectionner un pixel
pixel=img[50,100]

#selectionner une bande spectrale
band=img[:,:,5]

#charger l'image entière en mémoire cache (optimiser la performance algorithmique)
#arr = img.load()

#pour afficher l'image. Il faut lancer ipython dans le terminal avec la commande ipython --pylab
imshow(img,(1,2,3))
