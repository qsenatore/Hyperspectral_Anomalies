#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 14:05:26 2025

@author: senatorequentin
"""

import numpy as np
from matplotlib.pyplot import imshow
import cv2

from keras.preprocessing.image import img_to_array
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Cropping2D
from tensorflow.keras.models import Sequential

np.random.seed(31)

img_data = []

img = cv2.imread('Mbappe.jpg',1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

LIGNES = 582
COLONNES = 932

img_data.append(img_to_array(img))

img_array = np.reshape(img_data,(len(img_data),LIGNES,COLONNES,3))

img_array = img_array.astype('float32') / 255.

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(LIGNES, COLONNES, 3)))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
 

model.add(MaxPooling2D((2, 2), padding='same'))
     
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(3, (3, 3), activation='relu', padding='same', input_shape=(LIGNES, COLONNES, 3)))

model.add(Cropping2D(cropping=((1, 1), (2, 2))))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
model.summary()

model.fit(img_array, img_array, epochs = 50, shuffle = True)

pred = model.predict(img_array)

imshow(pred[0].reshape(LIGNES, COLONNES, 3))