import numpy as np
import scipy as sp
from scipy import ndimage
from scipy import signal
import matplotlib.pyplot as plt
import cv2


def gaussian2_xy(mean, cov, xy):
    invcov = np.linalg.inv(cov)
    results = np.ones([xy.shape[0], xy.shape[1]])
    for x in range(0, xy.shape[0]):
        for y in range(0, xy.shape[1]):
            v = xy[x,y,:].reshape(2,1) - mean
            results[x,y] = np.dot(np.dot(np.transpose(v), invcov), v)
    results = np.exp( - results / 2 )
    return results 

def gaussian2_n(mean, cov, n):
    s = int(n/2)
    x = np.linspace(-s,s,n)
    y = np.linspace(-s,s,n)
    xc, yc = np.meshgrid(x, y)
    xy = np.zeros([n, n, 2])
    xy[:,:,0] = xc
    xy[:,:,1] = yc

    return gaussian2_xy(mean, cov, xy), xc, yc

img_bgr = cv2.imread('lena.png')
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype('float32')
img_rgb = img_rgb/255.0

n = 111
mean =  np.array([0, 0])
mean = mean.reshape(2,1)
cov = np.array([[1,0],[0,1]])
g2d_kernel, xc, yc = gaussian2_n(mean, cov, 11)

blurImg = signal.convolve2d(img_rgb, g2d_kernel, mode='same', boundary='fill')
dx_kernel = np.array([[-1,0,1]])
dx = signal.convolve2d(blurImg, dx_kernel, mode='same', boundary='fill')

dy_kernel = np.array([[-1],[0],[1]])
dy = signal.convolve2d(blurImg, dy_kernel, mode='same', boundary='fill')

img_dmag = np.sqrt(dx**2 + dy**2)

thresh = sp.stats.threshold(img_dmag, threshmin=0.95, newval=0.0)
thresh = sp.stats.threshold(thresh, threshmax=0.1, newval=1.0)

#plt.figure(figsize=(10,10))
#plt.imshow(thresh)
#plt.show()

dx = signal.convolve2d(thresh, dx_kernel, mode='same', boundary='fill')
dy = signal.convolve2d(thresh, dy_kernel, mode='same', boundary='fill')

angles = np.arctan2(dx,dy)

bin_img = np.around(((angles * 57.2958 + 180) - 15)/30) # converted to degrees, shifted by 180 (so that none are negative) and then divided to equal # of bins in colour wheel

bin_img = bin_img.astype('uint8')

x,y = bin_img.shape

new_img = np.zeros((x,y,3))

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
                    [139, 32, 187],[244, 36, 148],[255, 32, 0],[255, 126, 2]]) # loops back to the colour at bin 0

print bin_img

for i in range (0,x):
    for j in range (0,y):
        if (thresh[i][j] != 0):
            colour_sel = bin_img[i][j]
            new_img[i][j][0] = colours[colour_sel][0]  ## Colours are shifted incorrectly. I'm not sure what needs to be changed.
            new_img[i][j][1] = colours[colour_sel][1]
            new_img[i][j][2] = colours[colour_sel][2]

plt.figure(figsize=(10,10))
new_img = new_img.astype('uint8')
new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
plt.imshow(new_img)
plt.show()