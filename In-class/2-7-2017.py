import argparse
import cv2
import numpy as np
import scipy as sp
from scipy import signal
from matplotlib import pyplot as plt

def gaussian1d(mu, sig, n):
    kernel = np.zeros([n]) 
    x = 0
    y = 0
    ksum = 0
    
    X = np.linspace(-(n/2),n/2, n)
    
    for i in range(0, n):
        holder = X[i]
        kernel[i] = np.exp(-((X[i] - mu) * (X[i] - mu)) / (2 * (sig*sig)))
        ksum += kernel[i]

    for i in range(0, n):
        kernel[i] = kernel[i] / ksum
    
    return kernel    

n = 3
kernel = gaussian1d([0],1,n)