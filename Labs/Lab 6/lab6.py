import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
import sys
import scipy as sp

def image_match(img_file1, img_file2):
    ##SIFT
    img1 = cv2.imread(img_file1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    src1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    
    sift1 = cv2.SIFT()
    src1 = np.uint8(src1)
    kp1 = sift1.detect(src1, None)
    kp1, des1 = sift1.compute(src1, kp1)
    
    ### 2
    
    img2 = cv2.imread(img_file2)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    src2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    
    sift2 = cv2.SIFT()
    src2 = np.uint8(src2)
    kp2 = sift2.detect(src2, None)
    kp2, des2 = sift2.compute(src2, kp2)
    ### Matching
    
    if (des1.shape[0] < des2.shape[0]):
        matches = np.zeros((des1.shape[0], des2.shape[0]))
        for i in range(0, des1.shape[0]):
            for j in range(0, des2.shape[0]):
                matches[i][j] = np.linalg.norm(des1[i]-des2[j])
    else: 
        matches = np.zeros((des2.shape[0], des1.shape[0]))
        for i in range(0, des2.shape[0]):
            for j in range(0, des1.shape[0]):
                matches[i][j] = np.linalg.norm(des2[i]-des1[j])
    
    gm = []
    threshold = 350
    
    for i in range(0, matches.shape[0]):
        #print min(matches[i][:])
        if (min(matches[i][:]) < threshold):
            gm.append(min(matches[i][:]))
        
    return len(gm)/float(matches.shape[0])

img_file1 = 'cn-tower-1s.jpg'
img_file2 = 'cn-tower-2s.jpg'

print "The image match score is " + str(image_match(img_file1, img_file2))
