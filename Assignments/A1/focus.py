import argparse
import cv2
import numpy as np
import time
import scipy as sp
from scipy import signal

def process_img(imgfile):
    start = time.time()
    print 'Opening ', imgfile
    img = cv2.imread(imgfile) 
    imgCol = img 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = cv2.convertScaleAbs(20 * np.log(np.abs(fshift)))
    
    rows, cols = img.shape
    crow,ccol = rows/2 , cols/2
    
    kernel = np.zeros((img.shape[0],img.shape[1]))

    kernel[crow][ccol] = 4
    kernel[crow - 1][ccol] = -1
    kernel[crow + 1][ccol] = -1
    kernel[crow][ccol - 1] = -1
    kernel[crow][ccol + 1] = -1
    
    f2 = np.fft.fft2(kernel)
    shiftedKernel = np.fft.fftshift(f2)
    magnitude_spectrum2 = cv2.convertScaleAbs(20 * np.log(np.abs(shiftedKernel)))
    
    fshift = fshift * shiftedKernel
    
    img_back = np.fft.ifft2(fshift)
    img_back = np.fft.ifftshift(img_back)
    img_back = np.abs(img_back)
    
    cv2.imshow('Derivative',cv2.convertScaleAbs(img_back))
    
    img_back = sp.stats.threshold(img_back, threshmin=50, newval=0.0)
    img_back = sp.stats.threshold(img_back, threshmax=1, newval=255.0)
    
    
    #cv2.imshow('Thresholded',cv2.convertScaleAbs(img_back))
    
    xStep = img.shape[0]/(img.shape[0] / 20)
    yStep = img.shape[1]/(img.shape[1] / 20)
        
    for i in range(1, (img.shape[0] / 20) + 1):
        for j in range (1, (img.shape[1] / 20) + 1):
            avg = np.sum(img_back[(xStep * (i - 1)) + 3:(xStep * i) + 3,(yStep * (j - 1)) + 3:(yStep * j) + 3]) / (xStep * yStep)
            if (avg > 0):
                cv2.rectangle(imgCol, (yStep * (j - 1) + 3,xStep * (i - 1) + 3), (yStep * j + 3,xStep * i + 3), (100,200,0), 2)
            
    cv2.imshow('Focused Regions',imgCol)
    
    stop = time.time()
    print "The Focus Analysis Took: {first}ms".format(first=(round((stop - start) * 1000,2)))
    print 'Press any key to proceed'   
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CSCI 4220U Lab 2.')
    parser.add_argument('--use-plotlib', action='store_true', help='If specified uses matplotlib for displaying images.')
    parser.add_argument('imgfile', help='Image file')
    args = parser.parse_args()
    
    process_img(args.imgfile)