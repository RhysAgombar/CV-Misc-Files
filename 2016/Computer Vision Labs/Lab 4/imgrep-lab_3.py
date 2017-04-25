# imgrep-lab.py
import argparse
import cv2
import numpy as np
import scipy as sp
from scipy import spatial
import matplotlib.pyplot as plt

def img_rep(imgfile):
    imgs = []
    means = []
    meanstds = []
    hists = []   
    
    img_bgr = cv2.imread(imgfile[0])
    img = cv2.cvtColor( img_bgr, cv2.COLOR_BGR2HSV )
        
    imgs.append( img )

    # Show the images.
    plt.figure()
    plt.title(imgfile[0])
    plt.imshow(imgs[0])
    
    img_bgr = cv2.imread(imgfile[0])
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB )

    mean, std = cv2.meanStdDev(img)
    means.append(mean)
    meanstds.append(np.concatenate([mean, std]).flatten())
    hists.append(cv2.calcHist(img, [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]).flatten())

    # Show the histograms
    centers = np.linspace(0, 512, 512)

    plt.figure()
    ax = plt.subplot(111)
    ax.bar(centers, hists[0], align='center', width=.5)
    plt.title("HSV 8x8x8")    
    
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

