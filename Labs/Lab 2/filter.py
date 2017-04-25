# filter.py
import argparse
import cv2
import numpy as np
import scipy as sp
from scipy import signal
from matplotlib import pyplot as plt

def gaussian2d(mu, cov, n):
    kernel = np.zeros([n,n,1])    
    ind = np.linspace(-(n/2),n/2, n)
    
    [X,Y] = np.meshgrid(ind,ind)
    
    x = 0
    y = 0
    ksum = 0
    for i in range(0, n):
        for j in range(0, n):
            holder = X[i,j]
            holder = Y[i,j]
            kernel[i,j] = np.exp(-((X[i,j] - mu[0]) * (X[i,j] - mu[0]) + (Y[i,j] - mu[1]) * (Y[i,j] - mu[1])) / (2 * (cov*cov)))
            ksum += kernel[i,j]

    for i in range(0, n):
        for j in range(0, n):
            kernel[i,j] = kernel[i,j] / ksum
    
    return kernel    

def filter(img):
    # Complete this method according to the tasks listed in the lab handout. 
    return img

def process_img1(imgfile):
    print 'Opening ', imgfile
    img = cv2.imread(imgfile)

    n = 3
    kernel = gaussian2d([0,0],1,n)
    kernel = kernel.reshape(n,n)

    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filtered_img = signal.convolve2d(grayImg, kernel, mode='same', boundary='fill')
    filtered_img = cv2.convertScaleAbs(filtered_img)
    
    # You should implement your functionality in filter function
    #filtered_img = filter(img)

    cv2.imshow('Input image',img)
    cv2.imshow('Filtered image',filtered_img)

    print 'Press any key to proceed'   
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_img2(imgfile):
    print 'Opening ', imgfile
    img = cv2.cvtColor(cv2.imread(imgfile), cv2.COLOR_RGB2BGR)
    
    kernel = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])

    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filtered_img = signal.convolve2d(grayImg, kernel, mode='same', boundary='fill')
    filtered_img = cv2.convertScaleAbs(filtered_img)

    plt.subplot(121)
    plt.imshow(img)
    plt.title('Input image')
    plt.xticks([]), plt.yticks([])

    plt.subplot(122)
    plt.imshow(filtered_img, cmap="Greys_r")
    plt.title('Filtered image')
    plt.xticks([]), plt.yticks([])

    plt.show()

def process_img3(imgfile):
    print 'Opening ', imgfile
    img = cv2.cvtColor(cv2.imread(imgfile), cv2.COLOR_RGB2BGR)
    
    kernel = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])

    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filtered_img = signal.convolve2d(grayImg, kernel, mode='same', boundary='fill')
    filtered_img = cv2.convertScaleAbs(filtered_img)

    plt.subplot(121)
    plt.imshow(img)
    plt.title('Input image')
    plt.xticks([]), plt.yticks([])

    plt.subplot(122)
    plt.imshow(filtered_img, cmap="Greys_r")
    plt.title('Filtered image')
    plt.xticks([]), plt.yticks([])

    plt.show()
    
def process_img4(imgfile):
    print 'Opening ', imgfile
    img = cv2.cvtColor(cv2.imread(imgfile), cv2.COLOR_RGB2BGR)
    
    xKernel = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    yKernel = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])

    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    filtered_imgX = signal.convolve2d(grayImg, xKernel, mode='same', boundary='fill')
    filtered_imgX = cv2.convertScaleAbs(filtered_imgX)
    
    filtered_imgY = signal.convolve2d(grayImg, yKernel, mode='same', boundary='fill')
    filtered_imgY = cv2.convertScaleAbs(filtered_imgY)

    intensity = np.sqrt(np.multiply(filtered_imgX, filtered_imgX) + np.multiply(filtered_imgY, filtered_imgY))

    plt.subplot(121)
    plt.imshow(img)
    plt.title('Input image')
    plt.xticks([]), plt.yticks([])

    plt.subplot(122)
    plt.imshow(intensity, cmap="Greys_r")
    plt.title('Filtered image')
    plt.xticks([]), plt.yticks([])

    plt.show()

def process_img5(imgfile):
    print 'Opening ', imgfile
    img = cv2.cvtColor(cv2.imread(imgfile), cv2.COLOR_RGB2BGR)
    
    kernel = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])

    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filtered_img = signal.convolve2d(grayImg, kernel, mode='same', boundary='fill')
    filtered_img = cv2.convertScaleAbs(filtered_img)

    plt.subplot(121)
    plt.imshow(img)
    plt.title('Input image')
    plt.xticks([]), plt.yticks([])

    plt.subplot(122)
    plt.imshow(filtered_img, cmap="Greys_r")
    plt.title('Filtered image')
    plt.xticks([]), plt.yticks([])

    plt.show()

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