import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from helper import grab_imgs, create_output_dir, normalize_img, output_starting_pairs, plot_results, plot_scores
import pca 
import outline
import dwt 
from imagemetrics import snr 
# Image Quality using Quality No Reference (QNR)
# grabbed from https://github.com/andrewekhalel/sewar
# from sewar import psnr


OUTPUT_DIR = 'Output/'
def arithmetic_analysis(img_list, title_list, output_dir, filename):
# Arithmetic Image Fusion
# Store as 3-tuple of (avg, min, max) for each pair
    alg_list = ['Mean', 'Min', 'Max']
    out_imgs = []
    for idx, (img1, img2) in enumerate(img_list):
        avg_img = np.mean([img1, img2], axis=0).astype(np.uint8)
        min_img = np.min([img1, img2], axis=0).astype(np.uint8)
        max_img = np.max([img1, img2], axis=0).astype(np.uint8)
        out_imgs.append((avg_img, min_img, max_img))
    output_file = output_dir + filename
    figsize = (5, 5)
    title='Arithmetic Methods'
    plot_results(out_imgs, alg_list, title_list, output_file, title, figsize)
    return out_imgs, alg_list
    # return 

def pca_analysis(img_list, title_list, output_dir, filename):
# PCA Fusion
    alg_list = ['PCA', 'PCA overlay']
    out_imgs = []
    for idx, (img1, img2) in enumerate(img_list):
        pca_fused1, pca_fused2 = pca.image_fusion(img1, img2, fusion_type='max_thresh', pct_components=0.95)
        out_imgs.append((pca_fused1, pca_fused2))
    output_file = output_dir + filename
    figsize = (4, 6)
    title = 'PCA Methods'
    plot_results(out_imgs, alg_list, title_list, output_file, title, figsize)
    return out_imgs, alg_list

def edge_detection_analysis(img_list, title_list, output_dir, filename):
# Edge Detection with Canny Edges
    alg_list = ['Canny', 'Fusion (Max)', 'Fusion (Max) +\n Smoothed']
    out_imgs = []
    for idx, (img1, img2) in enumerate(img_list):
        edges, fused_edges = outline.get_edges(img1, img2, smoothed=False)
        _, fused_edges_smooth = outline.get_edges(img1, img2, smoothed=True)
        out_imgs.append((edges, fused_edges, fused_edges_smooth))
    output_file = output_dir + filename 
    figsize = (5, 5)
    title = 'Edge Detection'
    plot_results(out_imgs, alg_list, title_list, output_file, title, figsize)
    return out_imgs, alg_list

def dwt_analysis(img_list, title_list, output_dir, filename):\
# Discrete Wavelet Transform Fusion
    alg_list = ['DWT (mean)', 'DWT (max)', 'DWT (min)']
    out_imgs = []
    for idx, (img1, img2) in enumerate(img_list):
        mean_img = dwt.image_fusion(img1, img2, method='mean')
        max_img = dwt.image_fusion(img1, img2, method='max')
        min_img = dwt.image_fusion(img1, img2, method='min')
        out_imgs.append((mean_img, max_img, min_img))
    output_file = output_dir + filename 
    figsize = (5, 5)
    title = 'Discrete Wavelet Transform (DWT)'
    plot_results(out_imgs, alg_list, title_list, output_file, title, figsize)
    return out_imgs, alg_list
    
def main():
    # Create output directory
    create_output_dir(OUTPUT_DIR)
    # Grab images, only grab LWIR and VIS images. IR or NIR in case of no LWIR
    tank1, tank2, _ = grab_imgs('tank')
    house1, _, house3 =  grab_imgs('house_3_men')
    _, kap2, kap3 =  grab_imgs('kaptein1123')
    soldier1, _, soldier3 = grab_imgs('soldier_behind_smoke')

    # Ideal results
    # Format (ideal_tank, ideal_house, ideal_kap, ideal smoke)
    ideal_tank, ideal_house, ideal_kap, ideal_soldier = grab_imgs('ideal')
    ideal_labels = ['Ideal Tank', 'Ideal House', 'Ideal Kaptein', 'Ideal Smoke']

    ideal_results = [((ideal_tank)), ((ideal_house)), ((ideal_kap)), ((ideal_soldier))]
    input_img_list = [
        (tank1, tank2),
        (house1, house3),
        (kap2, kap3),
        (soldier1, soldier3)
    ]
    title_list = [
        ('Tank LWIR', 'Tank VIS'),
        ('House LWIR', 'House VIS'),
        ('Kaptein IR', 'Kaptein VIS'),
        ('Smoke IR', 'Smoke VIS'),
    ]
    desc_list = ['Tank', 'House', 'Kaptein', 'Smoke']
    combined_imgs_out = []
    combined_labels_out = []
    # Generate starting images for the code before fusion begins
    # output_starting_pairs(img_list, title_list, OUTPUT_DIR, filename='starting_images.png', ideal_list = ideal_results)
    starting_imgs = [
        (tank1, house1, kap2, soldier1),
        (tank2, house3, kap3, soldier3),
        (ideal_tank, ideal_house, ideal_kap, ideal_soldier)
    ]
    starting_labels = ['LWIR / IR', 'Visible', 'Ideal Fusion']
    title = 'Starting Images'
    figsize = (7, 4)
    plot_results(starting_imgs, desc_list, starting_labels, OUTPUT_DIR + 'starting_images.png', title, figsize=figsize)
     
    # Generate Arithmetic image fusion analysis
    tmp_imgs, tmp_labels = arithmetic_analysis(input_img_list, desc_list, OUTPUT_DIR+'Arithmetic/', 'Arithmetic_Summary.png')
    combined_imgs_out.append(tmp_imgs)
    combined_labels_out.append(tmp_labels)

    # Gemerate PCA image fusion analysis
    tmp_imgs, tmp_labels = pca_analysis(input_img_list, desc_list, OUTPUT_DIR + 'PCA/', 'PCA_Summary.png')
    combined_imgs_out.append(tmp_imgs)
    combined_labels_out.append(tmp_labels)

    # Edge Detection image fusion analysis
    tmp_imgs, tmp_labels = edge_detection_analysis(input_img_list, desc_list, OUTPUT_DIR + 'Edge_Detection/', 'Edge_Summary.png')
    combined_imgs_out.append(tmp_imgs)
    combined_labels_out.append(tmp_labels)
    
    # Discrete Wavelet Transform (DWT) image fusion analysis
    tmp_imgs, tmp_labels = dwt_analysis(input_img_list, desc_list, OUTPUT_DIR + 'DWT/', 'DWT_Summary.png')
    combined_imgs_out.append(tmp_imgs)
    combined_labels_out.append(tmp_labels)

    # Add ideal results
    combined_imgs_out.append([[ideal_tank], [ideal_house], [ideal_kap], [ideal_soldier]])
    combined_labels_out.append(['Ideal Fusion'])

    # Combine results
    # Loop through the combined lists and reconstruct better format to use common functionality
    # print(np.unique(combined_imgs_out[5][0], return_counts=True))
    # print(qnr(np.expand_dims(input_img_list[0][0], input_img_list[0][1], input_img_list, combined_imgs_out[0][0]))
    # a = np.expand_dims(input_img_list[0][0], axis=2)
    # b = np.expand_dims(input_img_list[0][1], axis=2)
    # c = np.expand_dims(combined_imgs_out[0][0][0], axis=2)
    # print(c)
    # a = input_img_list[0][0]
    # b = input_img_list[0][1]
    # c = combined_imgs_out[0][0][0]
    # print(a.shape, b.shape, c.shape)
    # # print(qnr(a, b, c))
    # print(psnr(a, c), psnr(b, c))
    final_imgs = [[],[],[],[]]
    final_labels = []
    final_scores = [[],[],[],[]]
    for img_list, label_list in zip(combined_imgs_out, combined_labels_out):
        for idx in range(len(label_list)):
            for idy in range(len(img_list)):
                final_imgs[idy].append(img_list[idy][idx])
            final_labels.append(label_list[idx])
    output_file = OUTPUT_DIR + 'Methods_Summary.png'
    figsize = (18, 5)
    title = 'Methods Summary'
    plot_results(final_imgs, final_labels, desc_list, output_file, title, figsize)

    for idx in range(len(final_imgs)):
        for idy in range(len(final_imgs[idx])):
            final_scores[idx].append(snr(final_imgs[idx][idy]))
            if np.min(final_imgs[idx][idy]) < 0:
                print('negative')
            # psnr_score = psnr(final_imgs[idy][idx], input_img_list[idx][0])
            # psnr_score += psnr(final_imgs[idy][idx], input_img_list[idx][1])
            # final_scores[idx].append(psnr_score)

    # Image Quality using Quality No Reference (QNR)
    # grabbed from https://github.com/andrewekhalel/sewar
    
    # plt.clf()
    # Total Scores 
    plot_scores(final_scores, final_labels, desc_list, 'Algorithm Scores using SNR', 'SNR (dB)', OUTPUT_DIR + 'SNR_all.png', log=True)
    
    # print(avg_scores.shape)
    # for score in final_scores:
    #     avg = np.mean(score)
    #     avg_scores.append(avg)

    # Plot average scores
    avg_scores = np.mean(final_scores, axis=0)
    x = np.arange(len(final_labels))
    ax = plt.subplot(1, 1, 1)
    w = 0.6
    plt.xticks(x, final_labels, rotation=90)
    ax.bar(x + w/2, avg_scores, width=w, align='center', label=final_labels)
    plt.subplots_adjust(bottom=0.3)
    ax.set_ylabel('SNR (dB)')
    ax.set_title('Average Algorithm Scores using SNR')
    plt.savefig(OUTPUT_DIR + 'SNR_avg.png')
    # plt.show()
    # plot_scores(avg_scores, final_labels, desc_list, 'SNR (dB)', 'Signal-To-Noise(SNR) Avg Scores', OUTPUT_DIR + 'SNR_avg.png', log=True)

    return


if __name__ == '__main__':
    main()