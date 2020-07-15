import cv2 as cv
import numpy as np

# Possible improvement, using contours somehow
# def getContours(img):
#     contours, hierarchy = cv.findContours(img,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
#     for cnt in contours:
#         area = cv.contourArea(cnt)
#         cv.drawContours(IF,cnt,-1,(255,255,255),1)
#     return 

def get_edges(img1, img2, smoothed=True):
    if smoothed:
        img_blur = cv.GaussianBlur(img1, (7, 7), 0)
        edges = cv.Canny(img_blur, 100, 200)
    # print(np.unique(edges, return_counts=True))
        out_img = np.max([img2, edges], axis=0)
        return edges, out_img
    else:
        edges = cv.Canny(img1, 100, 200)
        out_img = np.max([img2, edges], axis=0)
        return edges, out_img

# I2, I1, IF = grab_imgs('kaptein1123')
# # I1 = cv.imread("resources/Use/ir.bmp")
# # I2 = cv.imread("resources/Use/ll.bmp")

# # IF = cv.imread("resources/Use/ll.bmp")

# kern = np.ones((5,5),np.uint8)

# IBlur = cv.GaussianBlur(I1,(7,7),0)
# IOutL = cv.Canny(IBlur, 300,300)
# IDial = cv.dilate(IOutL,kern,iterations=0)

# getContours(IDial)

# # imgStack = stackImages(0.6,([I1,I2,IBlur],
# #                             [IOutL,IDial,IF]))

# cv.imshow("contrast",I1)
# cv.imshow("other",I2)
# cv.imshow("out",IOutL)
# cv.imshow("Dia",IDial)
# # getContours(IF)
# cv.imshow("LINES",IF)
# # out = np.max([IF, ])
# cv.waitKey(0)
