import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import signal

img = cv2.imread("test.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#dx = np.array([1,-1])
#dx = dx.reshape(2,1)
#dy = dx.T

print img.shape

Ix = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
Iy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
#Iy = signal.convolve2d(img, dy, mode='same', boundary='fill')



cv2.imshow("test", Ix)
