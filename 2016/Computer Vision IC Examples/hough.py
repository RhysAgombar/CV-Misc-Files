import argparse
import cv2
import numpy as np
import scipy as sp
from scipy import signal
import matplotlib.pyplot as plt

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




im = cv2.imread('road.jpg')

#cv2.imshow('test',im)

im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype('float32')
im = im/255.0

n = 111
mean =  np.array([0, 0])
mean = mean.reshape(2,1)
cov = np.array([[1,0],[0,1]])
g2d_kernel, xc, yc = gaussian2_n(mean, cov, 11)

blurImg = signal.convolve2d(im, g2d_kernel, mode='same', boundary='fill')

dx_kernel = np.array([[-1,1]])
dx = signal.convolve2d(blurImg, dx_kernel, mode='same', boundary='fill')

dy_kernel = np.array([[-1],[1]])
dy = signal.convolve2d(blurImg, dy_kernel, mode='same', boundary='fill')

img_dmag = np.sqrt(dx**2 + dy**2)

dx = sp.stats.threshold(dx, threshmin=0.5, newval=0.0)
dx = sp.stats.threshold(dx, threshmax=0.5, newval=1.0)

dy = sp.stats.threshold(dy, threshmin=0.5, newval=0.0)
dy = sp.stats.threshold(dy, threshmax=0.5, newval=1.0)

angles = np.arctan(dx,dy)

cv2.imshow('test',angles)


x,y = angles.shape

print x,y

Nimg = np.zeros([x,y])

for i in range (0,x):
    for j in range (0,y):
        if (angles[i][j] != 0.0):
            print angles[i][j]
            #slope = -x
            #intercept = y
            cv2.line(Nimg, [0,j], (-i)*x, (255,0,0)) 

cv2.imshow('lin',Nimg)

#plt.figure(figsize=(10,10))
#plt.imshow(angles)
#plt.title('angles')
#plt.show()






#lines = cv2.HoughLines(angles, rho, theta, threshold) # Check OpenCV help to see how to use HoughLinesP
#for rho,theta in lines[0]:
#    a = np.cos(theta)
#   b = np.sin(theta)
#   x0 = a*rho
#   y0 = b*rho
#   x1 = int(x0 + 1000*(-b))
#   y1 = int(y0 + 1000*(a))
#   x2 = int(x0 - 1000*(-b))
#   y2 = int(y0 - 1000*(a))
#   cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
#cv2.imshow('houghlines',img)
#cv2.waitKey(0)