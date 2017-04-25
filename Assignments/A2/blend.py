import argparse
import cv2
import numpy as np
from scipy import signal
import math
from matplotlib import pyplot as plt

def makefilter(sup,sigma,tau):
  hsup=(sup-1)/2;
  
  ind = np.linspace(-hsup,hsup, sup)
  X,Y = np.meshgrid(ind, ind);
  
  r = np.power((np.multiply(X,X)+np.multiply(Y,Y)),0.5);
  
  f = np.cos(r * (math.pi * tau / sigma)) * np.exp(-(r*r)/(2*sigma*sigma))
  
  f=f-np.mean(f);
  f=f/np.sum(np.abs(f)); 
  
  return f

def process_img(img1N, img2N):   
    img1 = cv2.cvtColor(cv2.imread(img1N), cv2.COLOR_RGB2GRAY)
    img2 = cv2.cvtColor(cv2.imread(img2N), cv2.COLOR_RGB2GRAY)
      
    NF=13                 
    SUP=49                      
    F = np.zeros([SUP,SUP,NF])    
    F[:,:,0]=makefilter(SUP,2,1);
    F[:,:,1]=makefilter(SUP,4,1);
    F[:,:,2]=makefilter(SUP,4,2);
    F[:,:,3]=makefilter(SUP,6,1);
    F[:,:,4]=makefilter(SUP,6,2);
    F[:,:,5]=makefilter(SUP,6,3);
    F[:,:,6]=makefilter(SUP,8,1);
    F[:,:,7]=makefilter(SUP,8,2);
    F[:,:,8]=makefilter(SUP,8,3);
    F[:,:,9]=makefilter(SUP,10,1);
    F[:,:,10]=makefilter(SUP,10,2);
    F[:,:,11]=makefilter(SUP,10,3);
    F[:,:,12]=makefilter(SUP,10,4);
       
    filtered1 = signal.convolve2d(img1, F[:,:,0], mode='same', boundary='fill')
    filtered2 = signal.convolve2d(img2, F[:,:,0], mode='same', boundary='fill')
    
    hist1 = np.zeros([NF,256,1]) 
    hist1[0] = cv2.calcHist([filtered1.astype('uint8')],[0],None,[256],[0,256])     
    hist2 = np.zeros([NF,256,1]) 
    hist2[0] = cv2.calcHist([filtered2.astype('uint8')],[0],None,[256],[0,256])   
    
    for i in range(1,13):
       filtered1 = signal.convolve2d(img1, F[:,:,i], mode='same', boundary='fill')
       filtered2 = signal.convolve2d(img2, F[:,:,i], mode='same', boundary='fill')
       
       hist1[i] = cv2.calcHist([filtered1.astype('uint8')],[0],None,[256],[0,256])   
       hist2[i] = cv2.calcHist([filtered2.astype('uint8')],[0],None,[256],[0,256])          

    match = np.abs(hist1 - hist2)
    match = np.sum(match) / 255
    
    print match
    if (match < 2500):
        print "True"
    else:
        print "False"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CSCI 4220U Assignment 2.')
    parser.add_argument('--use-plotlib', action='store_true', help='If specified uses matplotlib for displaying images.')
    parser.add_argument('imgfile1', help='Image file')
    parser.add_argument('imgfile2', help='Image file')
    args = parser.parse_args()
    
    process_img(args.imgfile1, args.imgfile2)