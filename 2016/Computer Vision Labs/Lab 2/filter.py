# filter.py
import argparse
import cv2
import numpy as np
import scipy as sp
from scipy import signal
#from matplotlib import pyplot as plt

def filter(img):
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mu = np.array([0,0])
    mu = mu.reshape(2,1)
    cov = np.array([[20,0],[0,20]])
    n = 3

    filterArray = gaussian2d(mu,cov,n) ## giving it [x,y]
    
    filterArray_n = filterArray / np.sum(filterArray)
    
    grayBlur = signal.convolve2d(grayImg, filterArray_n, mode='same', boundary='fill')
    
    grayBlur = cv2.convertScaleAbs(grayBlur)
    
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
    img = cv2.imread(imgfile)

    # You should implement your functionality in filter function
    filtered_img = filter(img)

    cv2.imshow('Input image',img)
    cv2.imshow('Filtered image',filtered_img)

    print 'Press any key to proceed'   
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def filter2(img):
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    filterArray = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    
    grayBlur = signal.convolve2d(grayImg, filterArray, mode='same', boundary='fill')
    
    grayBlur = cv2.convertScaleAbs(grayBlur)
    
    return grayBlur

def process_img2(imgfile):
    print 'Opening ', imgfile
    img = cv2.imread(imgfile)

    # You should implement your functionality in filter function
    filtered_img = filter2(img)

    cv2.imshow('Input image',img)
    cv2.imshow('x-derivative',filtered_img)

    print 'Press any key to proceed'   
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def filter3(img):
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    filterArray = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    
    grayBlur = signal.convolve2d(grayImg, filterArray, mode='same', boundary='fill')
    
    grayBlur = cv2.convertScaleAbs(grayBlur)
    
    return grayBlur

def process_img3(imgfile):
    print 'Opening ', imgfile
    img = cv2.imread(imgfile)

    # You should implement your functionality in filter function
    filtered_img = filter3(img)

    cv2.imshow('Input image',img)
    cv2.imshow('y-derivative',filtered_img)

    print 'Press any key to proceed'   
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def filter4(img):
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    filterArr = np.array([[1,-1], [-1,1]])
    
    grayBlur = signal.convolve2d(grayImg, filterArr, mode='same', boundary='fill')
    
    grayBlur = cv2.convertScaleAbs(grayBlur)
    
    return grayBlur

def process_img4(imgfile):
    print 'Opening ', imgfile
    img = cv2.imread(imgfile)

    # You should implement your functionality in filter function
    filtered_img = filter4(img)

    cv2.imshow('Input image',img)
    cv2.imshow('I have no idea what the instructions mean.',filtered_img)

    print 'Press any key to proceed'   
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def filter5(img):
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    filterArray = np.array([[0,-0.25,0],[-0.25,1,-0.25],[0,-0.25,0]])
    
    grayBlur = signal.convolve2d(grayImg, filterArray, mode='same', boundary='fill')
    
    grayBlur = cv2.convertScaleAbs(grayBlur)
    
    return grayBlur

def process_img5(imgfile):
    print 'Opening ', imgfile
    img = cv2.imread(imgfile)

    # You should implement your functionality in filter function
    filtered_img = filter5(img)

    cv2.imshow('Input image',img)
    cv2.imshow('y-derivative',filtered_img)

    print 'Press any key to proceed'   
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#def process_img2(imgfile): gx = conv2(conv2(I,dx,'same'),s,'same');
#    print 'Opening ', imgfile
#    img = cv2.imread(imgfile)
#
#    # You should implement your functionality in filter function
#    filtered_img = filter(img)
#
#    # You should implement your functionality in filter function
#
#    plt.subplot(121)
#    plt.imshow(img)
#    plt.title('Input image')
#    plt.xticks([]), plt.yticks([])
#
#    plt.subplot(122)
#    plt.imshow(filtered_img)
#    plt.title('Filtered image')
#    plt.xticks([]), plt.yticks([])
#
#    plt.show()

#process_img4("C:/Users/100515147/Desktop/Computer Vision Labs/Lab 2/test.jpg")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CSCI 4220U Lab 2.')
    parser.add_argument('--use-plotlib', action='store_true', help='If specified uses matplotlib for displaying images.')
    parser.add_argument('--task', help='Task Number')
    parser.add_argument('imgfile', help='Image file')
    args = parser.parse_args()
    
    if args.task == '1':
        process_img1(args.imgfile)
    if args.task == '2':
        process_img2(args.imgfile)
    if args.task == '3':
        process_img3(args.imgfile)
    if args.task == '4':
        process_img4(args.imgfile)
    if args.task == '5':
        process_img5(args.imgfile)
