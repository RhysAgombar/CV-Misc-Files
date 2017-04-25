import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("test.jpg")
img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

mat1 = np.matrix([[1,2,1],[2,4,2],[1,2,1]])

##img3 = np.convolve(img2, mat1, 'same')

plt.figure(figsize=(17,7))
plt.imshow(img)

#xx, yy = np.mgrid[0:img1.shape[0], 0:img1.shape[1]]

#cv2.waitKey(0)