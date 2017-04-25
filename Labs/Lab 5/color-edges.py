import numpy as np
import scipy as sp
from scipy import ndimage
from scipy import signal
import matplotlib.pyplot as plt
import cv2

img_rgb = cv2.imread('lena.png')
##img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
##img_rgb = img_rgb/255.0


#0 = 255, 126, 2
#1 = 255, 168, 6
#2 = 255, 207, 1
#3 = 255, 248, 2
#4 = 133, 207, 0
#5 = 2, 184, 1
#6 = 0, 178, 214
#7 = 0, 122, 199
#8 = 1, 37, 187
#9 = 139, 32, 187
#10 = 244, 36, 148
#11 = 255, 32, 0
colours = np.array([[255, 126, 2],[255, 168, 6],[255, 207, 1],
                    [255, 248, 2],[133, 207, 0],[2, 184, 1],
                    [0, 178, 214],[0, 122, 199],[1, 37, 187],
                    [139, 32, 187],[244, 36, 148],[255, 32, 0],[255, 126, 2]])
                    
edges = cv2.Canny(img_rgb, 100, 200)

dx = ndimage.sobel(edges, axis=0, mode='constant')
dy = ndimage.sobel(edges, axis=1, mode='constant')
mag = np.sqrt(dx**2 + dy**2)

thresh = sp.stats.threshold(mag, threshmin=0.95, newval=0.0)
thresh = sp.stats.threshold(thresh, threshmax=0.1, newval=1.0)

cv2.imshow('thres', thresh)

angles = np.arctan2(dx,dy)

x,y = edges.shape
new_img = np.zeros((x,y,3))
for i in range (0,(x-1)):
    for j in range (0,(y-1)):
        if (angles[i][j] != 0):
            new_img[i][j] = colours[np.around((((angles[i][j] * 57.2958) + 180)- 15)/30)]
        
cv2.imshow('angles', new_img)

plt.imshow(new_img)
plt.show()
