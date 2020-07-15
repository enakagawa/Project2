# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 16:32:50 2020

@author: Owner
"""


import numpy as np
import pywt
import pywt.data
import matplotlib.pyplot as plt
import cv2

#reading in original images
i = cv2.imread(r"C:\Users\Owner\Desktop\navy\Project 2\Image dataset\TNO_Image_Fusion_Dataset\FEL_images\Nato_camp_sequence\thermal\1828i.bmp",0)
j = cv2.imread(r"C:\Users\Owner\Desktop\navy\Project 2\Image dataset\TNO_Image_Fusion_Dataset\FEL_images\Nato_camp_sequence\visual\1828v.bmp",0)

# We need to have both images the same size
I2 = cv2.resize(j,i.shape) # I do this just because i used two random images



# This function does the coefficient fusing according to the fusion method
def fuseCoeff(cooef1, cooef2, method):

    if (method == 'mean'):
        cooef = (cooef1 + cooef2) / 2
    elif (method == 'min'):
        cooef = np.minimum(cooef1,cooef2)
    elif (method == 'max'):
        cooef = np.maximum(cooef1,cooef2)
    else:
        cooef = []

    return cooef


# Params
FUSION_METHOD1 = 'max' # Can be 'min' || 'max || anything you choose according theory
FUSION_METHOD2 = 'min'



## Fusion algo
'''first image'''
#image decomposition
C1 = pywt.wavedec2(i,'haar',mode='periodization', level=2) #2 level DWT

'''second image'''
#image decomposition
C2 = pywt.wavedec2(j,'haar',mode='periodization', level=2) #2 level DWT

# Second: for each level in both image do the fusion according to the desire option
fusedCooef = []
for i in range(len(C1)-1):

    # The first values in each decomposition is the apprximation values of the top level
    if(i == 0):

        fusedCooef.append(fuseCoeff(C1[0],C2[0],FUSION_METHOD1))

    else:

        # For the rest of the levels we have tupels with 3 coeeficents
        c1 = fuseCoeff(C1[i][0], C2[i][0], FUSION_METHOD2)
        c2 = fuseCoeff(C1[i][1], C2[i][1], FUSION_METHOD2)
        c3 = fuseCoeff(C1[i][2], C2[i][2], FUSION_METHOD2)

        fusedCooef.append((c1,c2,c3))

# Third: After we fused the cooefficent we nned to transfor back to get the image
fusedImage = pywt.waverec2(fusedCooef, 'haar')

# Forth: normmalize values to be in uint8
fusedImage = np.multiply(np.divide(fusedImage - np.min(fusedImage),(np.max(fusedImage) - np.min(fusedImage))),255)
fusedImage = fusedImage.astype(np.uint8)

# Fith: Show image
cv2.imshow("1", i)
cv2.imshow('2', j)
cv2.imshow("win",fusedImage)
k = cv2.waitKey(0) #include otherwise kernel crashes
#press esc to not save, press s to save
if k==27:
    cv2.destroyAllWindows()