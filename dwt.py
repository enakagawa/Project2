# Code is based on the reconstruction found here:
# https://stackoverflow.com/questions/42608721/image-fusion-using-wavelet-transform-in-python
import numpy as np 
import pywt
import cv2 as cv 
from helper import grab_imgs, normalize_img

def fuse_coeff(partA, partB, method):
    if method == 'max':
        coef = np.maximum(partA, partB)
    elif method == 'mean':
        coef = (partA + partB) / 2
    elif method == 'min':
        coef = np.minimum(partA, partB)
    else:
        coef = []
    return coef

def image_fusion(img1, img2, wavelet='db1', method='mean'):
    # Wavelet can be adjusted
    # Method is one of either {'mean', 'max', 'min'}
    coeff1 = pywt.wavedec2(img1, wavelet)
    coeff2 = pywt.wavedec2(img2, wavelet)
    fusedCoeff = []
    
    for idx in range(len(coeff1)-1):
        if idx == 0:
            fusedCoeff.append(fuse_coeff(coeff1[0], coeff2[0], method))
        else:
            c1 = fuse_coeff(coeff1[idx][0], coeff2[idx][0], method)
            c2 = fuse_coeff(coeff1[idx][1], coeff2[idx][1], method)
            c3 = fuse_coeff(coeff1[idx][2], coeff2[idx][2], method)
            fusedCoeff.append((c1, c2, c3))
            
    fusedImage = pywt.waverec2(fusedCoeff, wavelet)
    # print('reached')
    # Normalize values 
    fusedImage = np.multiply(np.divide(fusedImage - np.min(fusedImage),(np.max(fusedImage) - np.min(fusedImage))),255)
    fusedImage = fusedImage.astype(np.uint8)
    return fusedImage

def main():
    # Soldier1, soldier3
    img1, _, img2 = grab_imgs('soldier_behind_smoke')

    # Attempt 1:

    wavelet = 'db1'
    fused_img = image_fusion(img1, img2, wavelet)
    cv.imshow('fused', fused_img)
    cv.waitKey(0)
    
    return 

if __name__ == '__main__':
    main()


# import pywt
# import cv2
# import numpy as np
# from helper import grab_imgs

# # This function does the coefficient fusing according to the fusion method
# def fuseCoeff(cooef1, cooef2, method):

#     if (method == 'mean'):
#         cooef = (cooef1 + cooef2) / 2
#     elif (method == 'min'):
#         cooef = np.minimum(cooef1,cooef2)
#     elif (method == 'max'):
#         cooef = np.maximum(cooef1,cooef2)
#     else:
#         cooef = []

#     return cooef


# # Params
# FUSION_METHOD = 'mean' # Can be 'min' || 'max || anything you choose according theory

# # Read the two image
# # I1 = cv2.imread('i1.bmp',0)
# # I2 = cv2.imread('i2.jpg',0)
# I1, _, I2 = grab_imgs('soldier_behind_smoke')

# # We need to have both images the same size
# # I2 = cv2.resize(I2,I1.shape) # I do this just because i used two random images

# ## Fusion algo

# # First: Do wavelet transform on each image
# wavelet = 'db1'
# cooef1 = pywt.wavedec2(I1[:,:], wavelet)
# cooef2 = pywt.wavedec2(I2[:,:], wavelet)

# # Second: for each level in both image do the fusion according to the desire option
# fusedCooef = []
# for i in range(len(cooef1)-1):

#     # The first values in each decomposition is the apprximation values of the top level
#     if(i == 0):

#         fusedCooef.append(fuseCoeff(cooef1[0],cooef2[0],FUSION_METHOD))

#     else:

#         # For the rest of the levels we have tupels with 3 coeeficents
#         c1 = fuseCoeff(cooef1[i][0],cooef2[i][0],FUSION_METHOD)
#         c2 = fuseCoeff(cooef1[i][1], cooef2[i][1], FUSION_METHOD)
#         c3 = fuseCoeff(cooef1[i][2], cooef2[i][2], FUSION_METHOD)

#         fusedCooef.append((c1,c2,c3))

# # Third: After we fused the cooefficent we nned to transfor back to get the image
# fusedImage = pywt.waverec2(fusedCooef, wavelet)

# # Forth: normmalize values to be in uint8
# fusedImage = np.multiply(np.divide(fusedImage - np.min(fusedImage),(np.max(fusedImage) - np.min(fusedImage))),255)
# fusedImage = fusedImage.astype(np.uint8)

# # Fith: Show image
# cv2.imshow("win",fusedImage)
# cv2.waitKey(0)