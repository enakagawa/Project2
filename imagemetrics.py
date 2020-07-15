# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 07:37:50 2020

@author: Owner
"""

import numpy as np
import cv2


def snr(img1):
    signal = np.mean(img1)
    noise = np.std(img1, axis = None)
    SNR = 10 * np.log10(signal/noise)
    return SNR 

if __name__ == '__main__':
    #reading in original image
    img1 = cv2.imread(r"",0)
    img2 = cv2.imread(r"",0)
        
    '''SNR''' #peak snr can be calculated below in the image quality index section
    signal=np.mean(img1)
    noise=np.std(img2 , axis = None)
    SNR=10*np.log(signal/noise)
    print ('\nSNR is printed in decibels \nAnything higher than 0 dB means more signal than noise \nHigher the better')
    print ("\nSNR for img1 : %.2f dB" %(SNR)) 

    '''image quality index'''
    #do pip install sewar in anaconda powershell terminal first
    #this has so many metrics already installed
    '''
    Mean Squared Error (MSE)
    Root Mean Sqaured Error (RMSE)
    Peak Signal-to-Noise Ratio (PSNR) [1]
    Structural Similarity Index (SSIM) [1]
    Universal Quality Image Index (UQI) [2]
    Multi-scale Structural Similarity Index (MS-SSIM) [3]
    Erreur Relative Globale Adimensionnelle de Synthèse (ERGAS) [4]
    Spatial Correlation Coefficient (SCC) [5]
    Relative Average Spectral Error (RASE) [6]
    Spectral Angle Mapper (SAM) [7]
    Spectral Distortion Index (D_lambda) [8]
    Spatial Distortion Index (D_S) [8]
    Quality with No Reference (QNR) [8]
    Visual Information Fidelity (VIF) [9]
    Block Sensitive - Peak Signal-to-Noise Ratio (PSNR-B) [10]
    '''
    from sewar.full_ref import qnr
    deformedimage=cv2.imread(r'',0)
    qualitynoref = qnr(deformedimage)