# imgrep-lab.py
import argparse
import cv2
import numpy as np
import scipy as sp
from scipy import spatial
import matplotlib.pyplot as plt
from scipy import signal

def img_rep(imgfile):
    imgs = []
    xFilter = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    yFilter = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])

    
    img_bgr = cv2.imread(imgfile[0])
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    grayImg = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    dx = signal.convolve2d(grayImg, xFilter, mode='same', boundary='fill')
    dy = signal.convolve2d(grayImg, yFilter, mode='same', boundary='fill')
    r = img_rgb[:,:,0]
    g = img_rgb[:,:,1]
    b = img_rgb[:,:,2]

    w, h, _ = img_bgr.shape

    img = np.zeros((w,h,5),dtype=np.uint8) #[[b],[g],[r],[dx],[dy]]

    img[:,:,0] = b
    img[:,:,1] = g
    img[:,:,2] = r
    img[:,:,3] = dx
    img[:,:,4] = dy

    #imgs.append( img )

    # Show the images.
    plt.figure()
    plt.title("Starry Night")
    plt.imshow(img_rgb)

    hists = []
    
    hists.append(cv2.calcHist(img, [0, 1, 2, 3, 4], None, [8, 8, 8, 8, 8], [0, 256, 0, 256, 0, 256, 0, 256, 0, 256]).flatten())

    # Show the histograms
    centers = np.linspace(0, 32768, 32768)

    plt.figure()
    ax = plt.subplot(111)
    ax.bar(centers, hists[0], align='center', width=.5)
    plt.title("8x8x8x8x8 Hist")    
    
    plt.show()
    
def load_image(imgfile):
    print 'Opening ', imgfile
    img = cv2.imread(imgfile)

    cv2.imshow('Input image',img)

    print 'Press any key to proceed'   
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    img_rep(['starry-night.jpg'])

