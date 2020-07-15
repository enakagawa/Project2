## README File 



### 1. Files
---

- Ideal_Ouputs/
- Output/
- TNO_Image_Fusion_Dataset/
- dwt/
- README.md
- dwt.py
- helper.py
- imagemetrics.py
- main.py
- outline.py
- pca.py 
- environment.yml

#### Folders:
- Ideal_Outputs/

    This folder consists our manually created ideal outputs using a photo editor 

- Output/

    This folder contains other subdirectories which contain the output image results in .png format. The images directly within this folder are different combined results that were found, including the Signal-To-Noise Ratio results, the starting images, and the summary of our results.

- TNO_Image_Fusion_Dataset/

    This folder contains the images used in our analysis gathered from [Figshare](https://figshare.com/articles/TNO_Image_Fusion_Dataset/1008029)

- dwt/

    This folder contains attempts at other DWT techniques


#### Files:
- README.md
    
    This File

- dwt.py

    This file contains the implementation of the Discrete Wavelet Transform using the PyWavelet (pywt) library. It contains the deconstruction and reconstruction based on a level 2 decomposition. It follows this [post](https://stackoverflow.com/questions/42608721/image-fusion-using-wavelet-transform-in-python). The fusion function is:
     
        image_fusion(img1, img2, wavelet='db1', method='mean')

        # Sample usage
        import dwt
        img1 = cv.imread('Some location')
        img2 = cv.imread('Other location')
        wavelet = 'db1'
        method = 'mean'     # Can be one of ['mean', 'max', 'min]
        fused_img = dwt.image_fusion(img1, img2, wavelet, method)


- helper.py

    This file contains helper methods for grabbing images, creating folders for output, and creating / saving plots and tables into **Output/**

    The process for grabbing images from the TNO dataset was pretty hard-coded with respect to the intended directories

        # Grabbing Images from TNO
        grab_imgs(name='tank')      # See helper.py for other possible datasets

        # Sample Usage:
        from helper import grab_imgs
        soldier1, soldier2, soldier3 = grab_imgs('soldier')

    The normalization function takes an image and normalizes the pixel values to range from 0-255

        # Normalization
        normalize_img(img)          # Returns an image normalized from 0-255

        # Sample Usage:
        from helper import normalize_img 
        img1 = cv.imread('Some Image')
        processed_img1 = some_processing_function(img1)
        normalized_result = normalize_img(procesed_img1)

    We don't describe the remaining methods as they are strictly used for plotting results using Matplotlib

- imagemetrics.py

    This file is for computing the signal-to-noise ratio (SNR). The main function that is used is shown below:

        # Calculate Signal-To-Noise (SNR)
        snr(img1)                   # Returns a score as the decibel value of the SNR

        # Sample Usage:
        from imagemetrics import snr 
        img1 = cv.imread('Some Image')
        snr_score = snr(img1)       # Returns the dB score of the SNR

- main.py

    This file runs the analysis on 4 datasets and outputs the results. It can be called on its own using:

        python main.py

- outline.py

    This file contains the canny edge detection from the opencv library. It calculates the edges in the IR / LWIR spectrum and then overlays the Visible image with it. 

        get_edges(IR_img, Visible_img, smoothed=True)    # Smoothed is a boolean indicating if we want to pre-smooth the image using a Gaussian Filter

        # Sample Usage:
        from outline import get_edges 
        img1 = cv.imread('Some Image')
        img2 = cv.imread('Other Image')
        edge_fused_img = get_edges(img1, img2, smoothed=False) 

- pca.py 

    This file contains the Principal Components Analysis Fusion using Scikit-Learn (sklearn). It takes two images, a *fusion_type* and a *pct* parameter. The *fusion_type* is used to indicate what sort of fusion we wish to use after reconstructing the two images using PCA. The *pct* parameter is used to indicate how much of the original image we want to reconstruct.

        image_fusion(img1, img2, fusion_type = 'max_thresh', pct=0.95)
        
        # Sample Usage
        import pca 
        img1 = cv.imread('Some Image')
        img2 = cv.imread('Other Image')
        fusion = 'max_thresh'               # One of ['max_thresh', 'max', 'min', 'mean']
        pct_keep = 0.8
        pca_fused_img = pca.image_fusion(img1, img2, fusion, pct_keep)

- environment.yml

    This file contains the conda environment specifications for the project. The environment is called *opencv*, but could easily be renamed to something more relevant like *image_fusion*. In general, the following packages were installed:

    * python 3.6
    * opencv 4.3.0
    * numpy 1.19.0
    * matplotlib 3.2.2
    * pillow 7.2.0
    * swear 0.4.3               (This one might not work the best)

    
### 2. Usage of Code

The code was run using a windows environment. The environment was managed using Anaconda. To download and then run the environment:

        conda env create -f environment.yml
        conda activate opencv

The environment is called *opencv*, but can be easily renamed by changing the first line in *environment.yml*. 


### 3. Other Potentially Useful Libraries
---

1. SciPy    (Library for scientific computing, contains some image processing)
2. Pandas   (useful for plotting alongside matplotlib)
