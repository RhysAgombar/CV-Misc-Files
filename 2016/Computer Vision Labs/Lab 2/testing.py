# filter.py
import argparse
import cv2
import numpy as np
import scipy as sp
from scipy import signal
#from matplotlib import pyplot as plt

def filter(img):
    # Complete this method according to the tasks listed in the lab handout. 
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mu = np.array([0,0])
    mu = mu.reshape(2,1)
    cov = np.array([[20,0],[0,20]])
    n = 3
    
    filterArray = gaussian2d(mu,cov,n) ## giving it [x,y]
    
    filterArray_n = filterArray / np.sum(filterArray)
    
    grayBlur = signal.convolve2d(grayImg, filterArray_n, mode='same', boundary='fill')
    
    return grayBlur
    
def gaussian2d(mu, cov, n):
    rng = int(n/2)
    x = np.linspace(-rng,rng,n)
    y = np.linspace(-rng,rng,n)
    
    grid = np.meshgrid(x,y)
    
    xy = np.zeros([n, n, 2])
    xy[:,:,0] = grid[0]
    xy[:,:,1] = grid[1]
    
    results = np.ones([xy.shape[0],xy.shape[1]])
    
    invcov = np.linalg.inv(cov)
    results = np.ones([xy.shape[0], xy.shape[1]])
    for x in range(0, xy.shape[0]):
        for y in range(0, xy.shape[1]):
            v = xy[x,y,:].reshape(2,1) - mu
            results[x,y] = np.dot(np.dot(np.transpose(v), invcov), v)
    results = np.exp( - results / 2 )
    
    return results

def process_img1(imgfile):
    print 'Opening ', imgfile
    img = cv2.imread('C:/Users/100515147/Desktop/Computer Vision Labs/Lab 2/test.jpg')
    # You should implement your functionality in filter function
    filtered_img = filter(img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.imshow('Input image',img)

    cv2.imshow('Filtered image',filtered_img)

    print 'Press any key to proceed'   
    cv2.waitKey(0)
    cv2.destroyAllWindows()

process_img1('test.jpg')

#import cv2
#import numpy as np
#from matplotlib import pyplot as plt
#
#n = 3
#cov = np.array([[20,0],[0,20]])
#mean = np.array([0,0])
#mean = mean.reshape(2,1)
#
#rng = int(n/2)
#x = np.linspace(-rng,rng,n)
#y = np.linspace(-rng,rng,n)
#
#grid= np.meshgrid(x,y)
#
#xy = np.zeros([n, n, 2])
#xy[:,:,0] = grid[0]
#xy[:,:,1] = grid[1]
#
#results = np.ones([xy.shape[0],xy.shape[1]])
#
#invcov = np.linalg.inv(cov)
#
#results = np.ones([xy.shape[0], xy.shape[1]])
#for x in range(0, xy.shape[0]):
#    for y in range(0, xy.shape[1]):
#        v = xy[x,y,:].reshape(2,1) - mean        
#        results[x,y] = np.dot(np.dot(np.transpose(v), invcov), v)
#
#results = np.exp( - results / 2 )
#
#print results