import numpy as np
import scipy as sp
from scipy import signal
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('messi.jpg', 0)

filtered_img = signal.convolve2d(img, img, mode='same', boundary='fill')

plt.imshow(filtered_img, 'gray')
plt.show()