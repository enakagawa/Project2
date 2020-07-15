# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 08:08:07 2020

@author: Abigail Christensen

This code performs a 1 level haar wavelet transform analysis of two images
"""

import numpy as np
import pywt
import pywt.data
import matplotlib.pyplot as plt
import cv2

#reading in original images
i = cv2.imread(r"C:\Users\Owner\Pictures\1491586637923.jpg",0)
j = cv2.imread(r"C:\Users\Owner\Pictures\1491586637923(2).jpg",0)

'''first image'''
#image decomposition
coeffs1 = pywt.dwt2(i,'haar',mode='periodization') #1 level DWT
cA, (cH, cV, cD) = coeffs1 #extracting coefficients 

#image reconstruction
nrec = pywt.idwt2(coeffs1, 'haar', mode='periodization') #1 level IDWT
nrec = np.uint8(nrec) #converting double to uint8 for display purposes

#displaying reconstructed image
plt.imshow(i, cmap=plt.cm.gray)
plt.title('original image')
plt.show()

#displaying decomposed images
plt.figure(figsize=(25,18))

plt.subplot(2,2,1)
plt.imshow(cA, cmap=plt.cm.gray)
plt.title('cA: Approximated Image')

plt.subplot(2,2,2)
plt.imshow(cH, cmap=plt.cm.gray)
plt.title('cH: Horizontal')

plt.subplot(2,2,3)
plt.imshow(cV, cmap=plt.cm.gray)
plt.title('cV: Vertical')

plt.subplot(2,2,4)
plt.imshow(cD, cmap=plt.cm.gray)
plt.title('cD: Diagonal')


plt.show()

#displaying reconstructed image
plt.imshow(nrec, cmap=plt.cm.gray)
plt.title('reconstructed image')
plt.show()

'''second image'''
#image decomposition
coeffs2 = pywt.dwt2(j,'haar',mode='periodization') #1 level DWT
cA2, (cH2, cV2, cD2) = coeffs2 #extracting coefficients 

#image reconstruction
nrec2 = pywt.idwt2(coeffs2, 'haar', mode='periodization') #1 level IDWT
nrec2 = np.uint8(nrec2) #converting double to uint8 for display purposes

#displaying reconstructed image
plt.imshow(j, cmap=plt.cm.gray)
plt.title('original image')
plt.show()

#displaying decomposed images
plt.figure(figsize=(25,18))

plt.subplot(2,2,1)
plt.imshow(cA2, cmap=plt.cm.gray)
plt.title('cA2: Approximated Image')

plt.subplot(2,2,2)
plt.imshow(cH2, cmap=plt.cm.gray)
plt.title('cH2: Horizontal')

plt.subplot(2,2,3)
plt.imshow(cV2, cmap=plt.cm.gray)
plt.title('cV2: Vertical')

plt.subplot(2,2,4)
plt.imshow(cD2, cmap=plt.cm.gray)
plt.title('cD2: Diagonal')


plt.show()

#displaying reconstructed image
plt.imshow(nrec2, cmap=plt.cm.gray)
plt.title('reconstructed image')
plt.show()


