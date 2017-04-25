import numpy as np
import scipy as sp
from scipy import signal
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('messi.jpg', 0) # Load in grayscale
cv2.imshow("before", img)

f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = cv2.convertScaleAbs(20 * np.log(np.abs(fshift)))

rows, cols = img.shape
crow,ccol = rows/2 , cols/2

kernel = np.zeros((img.shape[0],img.shape[1]))
kernel[crow][ccol + 1] = 1
kernel[crow - 1][ccol + 1] = 2
kernel[crow + 1][ccol + 1] = 1
kernel[crow][ccol - 1] = -1
kernel[crow - 1][ccol - 1] = -2
kernel[crow + 1][ccol - 1] = -1
    
f2 = np.fft.fft2(kernel)
shiftedKernel = np.fft.fftshift(f2)
magnitude_spectrum2 = cv2.convertScaleAbs(20 * np.log(np.abs(shiftedKernel)))

fshift = fshift * shiftedKernel

img_back = np.fft.ifft2(fshift)
img_back = np.fft.ifftshift(img_back)
img_back = cv2.convertScaleAbs(np.abs(img_back))

cv2.imshow("FFT", img_back)


kernel2 = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
convImg = signal.convolve2d(img, kernel2, mode='same', boundary='fill')
convImg = cv2.convertScaleAbs(convImg)

cv2.imshow("Convolved", convImg)