# sampling.py
import argparse
import cv2
import numpy as np
import scipy as sp
from scipy import ndimage
import time
import math

# Evaluates 2D Gaussian on xy grid.
# xy is two channel 2D matrix.  The first channel stores x coordinates
# [-1 0 1]
# [-1 0 1]
# [-1 0 1]
# and the second channel stores y coordinates
# [-1 -1 -1]
# [0   0  0]
# [1   1  1]
# So then we can pick out an xy value using xy[i,j,:].
# Check out gaussian2_n() to see how you can construct such
# an xy using numpy
#
# For gaussian2_xy() and gaussian2_n() methods
# mean is a 2x1 vector and cov is a 2x2 matrix
#
# Use the following code to generate a Gaussian kernel
# n = 111
# mean =  np.array([0, 0])
# mean = mean.reshape(2,1)
# cov = np.array([[3,0],[0,3]])
# g2d_kernel, xc, yc = gaussian2_n(mean, cov, 11)
#
# Plot Gaussian as follows
#
# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# surf = ax.plot_surface(xc, yc, g2d_kernel,rstride=1, cstride=1, cmap='coolwarm',
#                        linewidth=0, antialiased=False)

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

def make_laplacian_img(img):
    # To do
    
    dbg = 1
    
    mean = np.array([0,0])
    mean = mean.reshape(2,1)
    cov = np.array([[1,0],[0,1]])
    G, xc, yc = gaussian2_n(mean, cov, 5)
    
    dx = np.array([[-1,0,1]]) ## can't convolve a 2d with 1d. dimensions have to match. [[ ]] makes it 2d
    dx_G = sp.signal.covolve(G, dx, mode='same')
    ddx_G = sp.signal.covolve(dx_G, dx, mode='same') # Derivative
    
    dy = np.transpose(dx)
    dy_G = sp.signal.covolve(G, dy, mode='same')
    ddy_G = sp.signal.covolve(dy_G, dx, mode='same')
    
    L = ddx_G + ddy_G
    if dbg == 1:
        cv2.imshow('Gaussian',cv2.resize(G,(50,50)))
        cv2.imshow('Dx_G',cv2.resize(dx_G,(50,50)))
        cv2.imshow('Dy_G',cv2.resize(dy_G,(50,50)))
        cv2.imshow('Laplacian',cv2.resize(L,(50,50)))
        
    
    img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_f = img_g.astype('float32') / 255.0
    
    
    return sp.signal.convolve(img_f, L, mode='same')

def load_image(imgfile):
    print 'Opening ', imgfile
    img = cv2.imread(imgfile)

    laplacian_img = make_laplacian_img(img)
    
    cv2.imshow('Image', img)
    cv2.imshow('Laplacian (image)', laplacian_img)
    
    print 'Press any key to proceed'   
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CSCI 4220U Lab 2.')
    parser.add_argument('imgfile', help='Image file')
    args = parser.parse_args()

    load_image(args.imgfile)