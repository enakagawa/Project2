# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 10:32:22 2020

@author: Abigail Christensen

This code performs an n level haar wavelet transform analysis of two images

"""

import numpy as np
import pywt
import pywt.data
import matplotlib.pyplot as plt
import cv2

#reading in original images
i = cv2.imread(r"C:\Users\Owner\Desktop\navy\Project 2\Image dataset\TNO_Image_Fusion_Dataset\tank\LWIR.tif",0)
j = cv2.imread(r"C:\Users\Owner\Desktop\navy\Project 2\Image dataset\TNO_Image_Fusion_Dataset\tank\Vis.tif",0)


'''first image'''
#image decomposition
C1 = pywt.wavedec2(i,'haar',mode='periodization', level=2) #2 level DWT

C1rec = pywt.waverec2(C1,'haar',mode='periodization') 
C1rec = np.uint8(C1rec) #converting double to uint8 for display purposes

#extracting coefficients by level
C1A2 = C1[0]
(C1H2, C1V2, C1D2) = C1[-2]
(C1H1, C1V1, C1D1) = C1[-1]


'''Second image'''
#image decomposition
C2 = pywt.wavedec2(j,'haar',mode='periodization', level=2) #2 level DWT

C2rec = pywt.waverec2(C2,'haar',mode='periodization') 
C2rec = np.uint8(C2rec) #converting double to uint8 for display purposes


#extracting coefficients by level
C2A2 = C2[0]
(C2H2, C2V2, C2D2) = C2[-2]
(C2H1, C2V1, C2D1) = C2[-1]


'''recom'''
test = C1A2-C1H2-C1V2-C1D2
test = test/np.max(test)
test = cv2.resize(test, (test.shape[1]*3,test.shape[0]*3))
cv2.imshow("test",test)

test2 = C2A2+C2H2+C2V2+C2D2
test2 = test2/np.max(test2)
test2 = cv2.resize(test2, (test2.shape[1]*3,test2.shape[0]*3))
cv2.imshow("test2",test2)

k = cv2.waitKey(0) #include otherwise kernel crashes
#press esc to not save, press s to save
if k==27:
    cv2.destroyAllWindows()


'''image 1'''
#displaying original image
plt.imshow(i, cmap=plt.cm.gray)
plt.title('original image')
plt.show()

#displaying decomposed images
plt.figure(figsize=(25,18))

plt.subplot(2,2,1)
plt.imshow(C1A2, cmap=plt.cm.gray)
plt.title('Approximated Image: Level 2')

#C1A2= np.expand_dims(C1A2, axis=2)
#print(C1A2.shape)
#cv2.imshow('test',C1A2)
#cv2.waitKey(0)

plt.subplot(2,2,2)
plt.imshow(C1H2, cmap=plt.cm.gray)
plt.title('Horizontal Coeff: Level 2')


plt.subplot(2,2,3)
plt.imshow(C1V2, cmap=plt.cm.gray)
plt.title('Vertical Coeff: Level 2')

plt.subplot(2,2,4)
plt.imshow(C1D2, cmap=plt.cm.gray)
plt.title('Diagonal Coeff: Level 2')

plt.show()

arr, coeff_slices = pywt.coeffs_to_array(C1)
plt.figure(figsize=(20,20))
plt.imshow(arr, cmap=plt.cm.gray)
plt.title('All wavelet coeffs. up to level 2')

#displaying reconstructed image
plt.figure()
plt.imshow(C1rec, cmap=plt.cm.gray)
plt.title('Reconstructed Image')
plt.show()

'''image 2'''
#displaying reconstructed image
plt.imshow(j, cmap=plt.cm.gray)
plt.title('original image')
plt.show()

#displaying decomposed images
plt.figure(figsize=(25,18))

plt.subplot(2,2,1)
plt.imshow(C2A2, cmap=plt.cm.gray)
plt.title('Approximated Image: Level 2')

plt.subplot(2,2,2)
plt.imshow(C2H2, cmap=plt.cm.gray)
plt.title('Horizontal Coeff: Level 2')

plt.subplot(2,2,3)
plt.imshow(C2V2, cmap=plt.cm.gray)
plt.title('Vertical Coeff: Level 2')

plt.subplot(2,2,4)
plt.imshow(C2D2, cmap=plt.cm.gray)
plt.title('Diagonal Coeff: Level 2')

plt.show()

arr, coeff_slices = pywt.coeffs_to_array(C2)
plt.figure(figsize=(20,20))
plt.imshow(arr, cmap=plt.cm.gray)
plt.title('All wavelet coeffs. up to level 2')

#displaying reconstructed image
plt.figure()
plt.imshow(C2rec, cmap=plt.cm.gray)
plt.title('Reconstructed Image')
plt.show()



        
        
