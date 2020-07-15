import os 
import numpy as np
import cv2 as cv 
import matplotlib.pyplot as plt 
from PIL import Image 

# Image grabbing
def grab_imgs(name='tank'):
    PARENT_DIR = os.path.join(os.getcwd(), ".\TNO_Image_Fusion_Dataset")
    if name == 'tank':
        img1_filepath = os.path.join(PARENT_DIR, 'tank', 'LWIR.tif')
        img1 = cv.imread(img1_filepath, cv.IMREAD_GRAYSCALE)
        img2_filepath = os.path.join(PARENT_DIR, 'tank', 'Vis.tif')
        img2 = cv.imread(img2_filepath, cv.IMREAD_GRAYSCALE)
        # Output in form (LWIR, VIS)
        return (img1, img2, None)

    if name == 'bunker':
        img1_filepath = os.path.join(PARENT_DIR, 'Athena_images', 'bunker', 'IR_bunker_g.bmp')
        img2_filepath = os.path.join(PARENT_DIR, 'Athena_images', 'bunker', 'bunker_r.bmp')
        img3_filepath = os.path.join(PARENT_DIR, 'Athena_images', 'bunker', 'VIS_bunker-rg.bmp')
        img1 = cv.imread(img1_filepath, cv.IMREAD_GRAYSCALE)
        img2 = cv.imread(img2_filepath, cv.IMREAD_GRAYSCALE)
        img3 = cv.imread(img3_filepath, cv.IMREAD_GRAYSCALE)
        # Output in form (Red, IR (Green), VIS (Red / Green))
        return (img1, img2, img3)

    
    if name == 'sandpath':
        img1_filepath = os.path.join(PARENT_DIR, 'DHV_images', 'sandpath', 'IR_18rad.bmp')
        img2_filepath = os.path.join(PARENT_DIR, 'DHV_images', 'sandpath', 'NIR_18dhvG.bmp')
        img3_filepath = os.path.join(PARENT_DIR, 'DHV_images', 'sandpath', 'VIS_18dhvR.bmp')
        img1 = cv.imread(img1_filepath, cv.IMREAD_GRAYSCALE)
        img2 = cv.imread(img2_filepath, cv.IMREAD_GRAYSCALE)
        img3 = cv.imread(img3_filepath, cv.IMREAD_GRAYSCALE)
        # Output in form (IR, NIR, VIS)
        return (img1, img2, img3)

    if name == 'Nato_camp':
        img1_filepath = os.path.join(PARENT_DIR, 'FEL_images', 'Nato_camp_sequence', 'thermal', '1807i.bmp')
        img2_filepath = os.path.join(PARENT_DIR, 'FEL_images', 'Nato_camp_sequence', 'visual', '1807v.bmp')
        img1 = cv.imread(img1_filepath, cv.IMREAD_GRAYSCALE)
        img2 = cv.imread(img2_filepath, cv.IMREAD_GRAYSCALE)
        # Output in form (IR, VIS)
        return (img1, img2, None)

    if name == 'soldier_behind_smoke':
        img1_filepath = os.path.join(PARENT_DIR, 'Athena_images', 'soldier_behind_smoke_1', 'IR_meting012-1200_g.bmp')
        img2_filepath = os.path.join(PARENT_DIR, 'Athena_images', 'soldier_behind_smoke_1', 'meting012-1200_rg.bmp.bmp')
        img3_filepath = os.path.join(PARENT_DIR, 'Athena_images', 'soldier_behind_smoke_1', 'VIS_meting012-1200_r.bmp')
        img1 = cv.imread(img1_filepath, cv.IMREAD_GRAYSCALE)
        img2 = cv.imread(img2_filepath, cv.IMREAD_GRAYSCALE)
        img3 = cv.imread(img3_filepath, cv.IMREAD_GRAYSCALE)
        # Output in form (IR, IR / RG, VIS)
        return (img1, img2, img3)
    
    if name == 'bench':
        img1_filepath = os.path.join(PARENT_DIR, 'DHV_images', 'bench', 'IR_37rad.bmp')
        img2_filepath = os.path.join(PARENT_DIR, 'DHV_images', 'bench', 'NIR_37dhvG.bmp')
        img3_filepath = os.path.join(PARENT_DIR, 'DHV_images', 'bench', 'VIS_37dhvR.bmp')
        img1 = cv.imread(img1_filepath, cv.IMREAD_GRAYSCALE)
        img2 = cv.imread(img2_filepath, cv.IMREAD_GRAYSCALE)
        img3 = cv.imread(img3_filepath, cv.IMREAD_GRAYSCALE)
        # Output in form (IR, NIR, VIS)
        return (img1, img2, img3)

    if name == 'house_3_men':
        img1_filepath = os.path.join(PARENT_DIR, 'Triclobs_images', 'houses_with_3_men', 'LWIR.bmp')
        img2_filepath = os.path.join(PARENT_DIR, 'Triclobs_images', 'houses_with_3_men', 'NIR.bmp')
        img3_filepath = os.path.join(PARENT_DIR, 'Triclobs_images', 'houses_with_3_men', 'VIS.bmp')
        img1 = cv.imread(img1_filepath, cv.IMREAD_GRAYSCALE)
        img2 = cv.imread(img2_filepath, cv.IMREAD_GRAYSCALE)
        img3 = cv.imread(img3_filepath, cv.IMREAD_GRAYSCALE)
        # Output in form (LWIR, NIR, VIS)
        return (img1, img2, img3)
    
    if name == 'kaptein1123':
        img1_filepath = os.path.join(PARENT_DIR, 'Triclobs_images', 'Kaptein_1123', 'Kaptein_1123_II.bmp')
        img2_filepath = os.path.join(PARENT_DIR, 'Triclobs_images', 'Kaptein_1123', 'Kaptein_1123_IR.bmp')
        img3_filepath = os.path.join(PARENT_DIR, 'Triclobs_images', 'Kaptein_1123', 'Kaptein_1123_VIS.bmp')
        img1 = cv.imread(img1_filepath, cv.IMREAD_GRAYSCALE)
        img2 = cv.imread(img2_filepath, cv.IMREAD_GRAYSCALE)
        img3 = cv.imread(img3_filepath, cv.IMREAD_GRAYSCALE)
        # Output in form (II, IR, VIS)
        return (img1, img2, img3)
    if name == 'ideal':
        PARENT_DIR = os.path.join(os.getcwd(), ".\Ideal_Outputs")
        img1_filepath = os.path.join(PARENT_DIR, 'tank.png')
        img2_filepath = os.path.join(PARENT_DIR, 'houses_with_3_men.png')
        img3_filepath = os.path.join(PARENT_DIR, 'kaptein_1123.png')
        img4_filepath = os.path.join(PARENT_DIR, 'soldier_behind_smoke_1.png')
        img1 = cv.imread(img1_filepath, cv.IMREAD_GRAYSCALE)
        img2 = cv.imread(img2_filepath, cv.IMREAD_GRAYSCALE)
        img3 = cv.imread(img3_filepath, cv.IMREAD_GRAYSCALE)
        img4 = cv.imread(img4_filepath, cv.IMREAD_GRAYSCALE)
        
        return (img1, img2, img3, img4)

# Creating output directories
def create_output_dir(out_dir='Output/', subdirs=['PCA/', 'Arithmetic/', 'DWT/', 'MHWT/', 'Edge_Detection/']):
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, out_dir)
    # analysis = ['PCA/', 'Arithmetic/', 'DWT/', 'MHWT/', 'Edge_Detection/']
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    for subdir in subdirs:
        temp_dir = os.path.join(results_dir, subdir)
        if not os.path.isdir(temp_dir):
            os.makedirs(temp_dir)   
    return

#  Image processing techniques
def normalize_img(img):
    img_out = (img - np.min(img))
    img_out = img_out / img_out.max()
    img_out *= 255
    img_out = img_out.astype(np.uint8)
    # img_out = cv.normalize(img, None, 255, 0)
    return img_out


# Plot starting images in grid (title space in between rows is larger than in following method)
def output_starting_pairs(img_list, title_list, output_dir, filename, ideal_list = None):
    fig, axs = plt.subplots(len(img_list[0]), len(img_list), figsize=(8, 4))
    if ideal_list != None:
        fig, axs = plt.subplots(len(img_list[0]) + 1, len(img_list), figsize=(8, 6))
    for idx, ((img1, img2), (title1, title2)) in enumerate(zip(img_list, title_list)):
        axs[0, idx].imshow(Image.fromarray(img1).convert('LA'))
        axs[1, idx].imshow(Image.fromarray(img2).convert('LA'))
        # axs[2, idx].imshow(Image.fromarray(img3).convert('LA'))
        axs[0, idx].set_title(title1)
        axs[1, idx].set_title(title2)
        axs[0, idx].set_yticks([])        
        axs[1, idx].set_yticks([])        
        # axs[2, idx].set_yticks([])
        axs[0, idx].set_xticks([])
        axs[1, idx].set_xticks([])        
        # axs[2, idx].set_xticks([])
    if ideal_list != None:
        for idx in range(len(ideal_list)):
            axs[2, idx].imshow(Image.fromarray(ideal_list[idx]).convert('LA'))
            axs[2, idx].set_yticks([])
            axs[2, idx].set_xticks([])
            axs[2, idx].set(xlabel=title_list[idx][0])
        axs[2, 0].set(ylabel='Ideal Fusion')
        # print((axs[2].shape))
        # axs[2].set_title('Ideal Results')
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    filename = output_dir + filename
    plt.savefig(filename)
    plt.close()
    return 

# Plot results nicely in tabulated format
def plot_results(img_list, x_labels, y_labels, filename, title, figsize):
    n_rows = len(img_list)
    n_cols = len(img_list[0])
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
    for idx in range(n_rows):
        for idy in range(n_cols):
            axs[idx, idy].imshow(Image.fromarray(img_list[idx][idy]).convert('LA'))
            axs[idx, idy].set_yticks([])
            axs[idx, idy].set_xticks([])
            if idy == 0:
                axs[idx, idy].set(ylabel=y_labels[idx])
            if idx == n_rows - 1:
                axs[idx, idy].set(xlabel=x_labels[idy])
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    # plt.title.set_position([0.5, 0.95])
    fig.suptitle(title, y=0.95)
    # filename = output_dir + filename
    plt.savefig(filename)
    plt.close()
    return
    
def plot_scores(score_list, labels, legend, title, error_type, filename, log=False):
    x = np.arange(len(labels))*2
    ax = plt.subplot(1, 1, 1)
    w = 0.4
    plt.xticks(x + w/4, labels, rotation=90)
    for idx, score in enumerate(score_list):
        pop = ax.bar(x+(idx-1)*w, score, width=w, align='center', label=legend[idx])
    plt.subplots_adjust(bottom=.45)
    ax.set_ylabel(error_type)
    ax.set_xlabel('Fusion Methods')
    ax.set_title(title)
    ax.legend()
    # if log:
    #     # plt.yscale('log')
    #     pass
    # ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.savefig(filename)
    plt.close()
    pass