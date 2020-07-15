import cv2 as cv 
import numpy as np 
from sklearn.decomposition import PCA 
from helper import normalize_img 

def image_fusion(img1, img2, fusion_type = 'max_thresh', pct_components=0.95):
# PCA image fusion
# https://stats.stackexchange.com/questions/229092/how-to-reverse-pca-and-reconstruct-original-variables-from-several-principal-com
    pca1 = PCA(n_components=pct_components, svd_solver='full')
    pca1.fit(img1)
    P1_transformed = pca1.fit_transform(img1)
    P1 = pca1.inverse_transform(P1_transformed)     # P1 is the reconstructed PCA image of img1 using {pct_components}% reconstruction

    pca2 = PCA(n_components=pct_components, svd_solver='full')
    pca2.fit(img2)
    P2_transformed = pca2.fit_transform(img2)
    P2 = pca2.inverse_transform(P2_transformed)     # P2 is the reconstructed PCA image of img2 using {pct_components}% reconstruction
    
    if fusion_type == 'max_thresh':
        _, P1_thresholded = cv.threshold(P1, 0.5, 1, cv.THRESH_BINARY)
        _, P2_thresholded = cv.threshold(P2, 0.5, 1, cv.THRESH_BINARY)
        fused1 = normalize_img(np.max([P1, P2], axis=0))
        fused2 = normalize_img(np.max([P1, normalize_img(img2)], axis=0))
        return fused1, fused2
    
    if fusion_type == 'max':
        fused1 = normalize_img(np.max([P1, P2], axis=0))
        fused2 = normalize_img(np.max([P1, normalize_img(img2)], axis=0))
        return fused1, fused2

    if fusion_type == 'mean':
        fused1 = normalize_img(np.mean([P1, P2], axis=0))
        fused2 = normalize_img(np.mean([P1, normalize_img(img2)], axis=0))
        return fused1, fused2
    
    if fusion_type == 'min':
        fused1 = normalize_img(np.min([P1, P2], axis=0))
        fused2 = normalize_img(np.min([P1, normalize_img(img2)], axis=0))
        return fused1, fused2
        
    return None, None 