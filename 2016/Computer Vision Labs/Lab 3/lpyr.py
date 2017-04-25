# lpyr.py
import argparse
import cv2
import numpy as np
from scipy import signal

def cvtToGrayAndBlur(img):
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    mu = np.array([0,0])
    mu = mu.reshape(2,1)
    cov = np.array([[20,0],[0,20]])
    
    n = 11

    filterArray = gaussian2d(mu,cov,n)
    
    filterArray_n = filterArray / np.sum(filterArray)

    grayBlur = signal.convolve2d(grayImg, filterArray_n, mode='same', boundary='fill')
    grayBlur = cv2.convertScaleAbs(grayBlur)
    return grayImg, grayBlur

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

def make_lpyr(img, minsize):
    grayImg, grayBlur = cvtToGrayAndBlur(img)
    
    w,h,c = img.shape
    w1 = w
    h1 = h
    i = 0

    if (w > h):
        while(h1>= 128):
            h1 = h1/2
            i = i + 1
    else:
        while(w1>= 128):
            w1 = w1/2
            i = i + 1

    lpyr = np.zeros((w,h,i)).astype(np.uint8)
    i = 0
    lap = np.subtract(grayImg, grayBlur)
    lpyr[:,:,0] = lap
 
    while(w/2 >= 128 and h/2 >= 128):
        w = w/2
        h = h/2
        re_gray = cv2.resize(grayImg, (h,w))
        re_blur = cv2.resize(grayBlur, (h,w))
        i = i + 1
        lap = np.subtract(re_gray, re_blur)
        lap = cv2.convertScaleAbs(lap)
        lpyr[0:lap.shape[0],0:lap.shape[1],i] = lap    
 
    return lpyr

def process_img(imgfile):
    #imgfile = 'C:/Users/100515147/Desktop/Computer Vision Labs/Lab 3/capture.jpg'
    print 'Opening ', imgfile
    img = cv2.imread(imgfile)
    cv2.imshow('img', img)
    print 'print any key'
    cv2.waitKey(0)

    # Complete this function
    #
    # This function converts the image into grayscale
    # and constructs a Laplacian pyramid from this image.
    # The smallest size of the image in the pyramid is 
    # determined by minsize argument
    
    ## Laplacion = image - blurred image
    # have to be >128 pixels
    
    lpyr = make_lpyr(img, minsize=128)
    w,h,c = lpyr.shape
    for i in range(0,c):
        cv2.imshow('image: ' + str(i), lpyr[:,:,i].reshape((w,h)))
        print 'press any key'
        cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CSCI 4220U Lab 3.')
    parser.add_argument('imgfile', help='Image file')
    args = parser.parse_args()

    process_img(args.imgfile)