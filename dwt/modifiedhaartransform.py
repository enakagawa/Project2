# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 07:58:48 2020

@author: Owner
"""


import numpy as np
import pywt
import pywt.data
import matplotlib.pyplot as plt
import cv2
import math

#reading in original images
img1 = cv2.imread(r"C:\Users\Owner\Desktop\navy\Project 2\Image dataset\TNO_Image_Fusion_Dataset\FEL_images\Nato_camp_sequence\thermal\1828i.bmp",0)
img2 = cv2.imread(r"C:\Users\Owner\Desktop\navy\Project 2\Image dataset\TNO_Image_Fusion_Dataset\FEL_images\Nato_camp_sequence\visual\1828v.bmp",0)
cv2.imshow("1", img1)
cv2.imshow('2', img2)

# We need to have both images the same size
I2 = cv2.resize(img2,img1.shape) # I do this just because i used two random images


# Fusion type
FUSION_METHOD = 'max'


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

#modified Haar transform definition
def modHaar(image):
    rownum = image.shape[0]
    colnum = image.shape[1]
    
    #looping through rows
    for i in range(0,rownum):
        for j in range(0,colnum):
            if (j<colnum/4):
                image[i][j]=(image[i][4*j-3]+image[i][4*j-2]+image[i][4*j-1]+image[i][4*j])/4
    for i in range(0,rownum):
        for j in range(0,colnum):
            if (j<colnum/4):
                image[i][math.ceil(colnum/4)+j]=((image[i][4*j-3]+image[i][4*j-2])-(image[i][4*j-1]+image[i][4*j]))/4
            else:
                if (math.ceil(colnum/4)+j)<colnum:
                    image[i][math.ceil(colnum/4)+j]=0     
    #looping through columns
    for j in range(0,colnum):
        for i in range(0,rownum):
            if (i<rownum/4):
                image[i][j]=(image[4*i-3][j]+image[4*i-2][j]+image[4*i-1][j]+image[4*i][j])/4
    for j in range(0,colnum):
        for i in range(0,rownum):
            if (i<rownum/4):
                image[math.ceil(rownum/4)+i][j]=((image[4*i-3][j]+image[4*i-2][j])-(image[4*i-1][j]+image[4*i][j]))/4
            else:
                if (math.ceil(rownum/4)+i)<rownum:
                    image[math.ceil(rownum/4)+i][j]=0 
    
 ''''   #removing columns and rows of all 0's (***section not finished***)
    for n in ()
    image = np.delete(image, 1, 0)'''
        
    return image

#implementing Modified Haar Wavelet Transform, then decomposing it using DWT            
new1 = modHaar(img1)
C1 = pywt.wavedec2(new1,'haar',mode='periodization', level=1) #1 level DWT for reconstruction purposes
new2 = modHaar(img2)
C2 = pywt.wavedec2(new2,'haar',mode='periodization', level=1) #1 level DWT for reconstruction purposes

# fusion according to the desire option
fusedCooef = []

for i in range(len(C1)-1):
    # The first values in each decomposition is the apprximation values of the top level
    if(i == 0):
        fusedCooef.append(fuseCoeff(C1[0],C2[0],FUSION_METHOD))
    else:
        # For the rest of the levels we have tupels with 3 coeeficents
        c1 = fuseCoeff(C1[i][0], C2[i][0], FUSION_METHOD)
        c2 = fuseCoeff(C1[i][1], C2[i][1], FUSION_METHOD)
        c3 = fuseCoeff(C1[i][2], C2[i][2], FUSION_METHOD)
        fusedCooef.append((c1,c2,c3))

# Third: After we fused the cooefficent we nned to transfor back to get the image
fusedImage = pywt.waverec2(fusedCooef, 'haar')

# Forth: normmalize values to be in uint8
fusedImage = np.multiply(np.divide(fusedImage - np.min(fusedImage),(np.max(fusedImage) - np.min(fusedImage))),255)
fusedImage = fusedImage.astype(np.uint8)

# Fith: Show image
cv2.imshow("win",fusedImage)
k = cv2.waitKey(0) #include otherwise kernel crashes
#press esc to not save, press s to save
if k==27:
    cv2.destroyAllWindows()