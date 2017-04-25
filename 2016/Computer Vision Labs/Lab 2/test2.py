import cv2
from cv2.cv import *
from cv2 import *
import numpy as np
import scipy as sp
from scipy import signal


img = LoadImage('C:/Users/100515147/Desktop/Computer Vision Labs/Lab 2/test.jpg')

# You should implement your functionality in filter function
filtered_img = img #filter(img)

cv2.imshow('Input image',img)
cv2.imshow('Filtered image',filtered_img)

print 'Press any key to proceed'   
cv2.waitKey(0)
cv2.destroyAllWindows()