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
    
    img_bgr = cv2.imread(imgfile[0])
    img = cv2.cvtColor( img_bgr, cv2.COLOR_BGR2RGB )
        
    imgs.append( img )

    # Show the images.
    plt.figure()
    plt.title(imgfile[0])
    plt.imshow(imgs[0])
    
    img_bgr = cv2.imread(imgfile[0])
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB )

    w,h,_ = img.shape
    
    w4 = w/4
    h4 = h/4

    nHist = []
    
    for i in range(0,4):
        for j in range(0,4):
            nHist.append( cv2.calcHist(img[w4*i:w4*(i+1),h4*j:h4*(j+1)], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]).flatten() )
            mean, std = cv2.meanStdDev(img[w4*i:w4*(i+1),h4*j:h4*(j+1)])
            means.append( mean )
            meanstds.append( np.concatenate([mean, std]).flatten() )

    # Show the histograms
    centers = np.linspace(0, 512, 512)

    plt.figure()

    for i in range(0,16):
        ax = plt.subplot(4,4,i+1)
        ax.bar(centers, nHist[i], align='center', width=.5)
        plt.title(("Subplot: ", i))
    

    # Now lets do some fancy distance computations.
    # Recall that we have successfully represented each image
    # as vectors.
    X_mean = []
    for m in means:
        X_mean = np.concatenate([X_mean, m.flatten()])
    X_mean = X_mean.reshape([16,3])
        
    X_meanstd = []
    for ms in meanstds:
        X_meanstd = np.concatenate([X_meanstd, ms.flatten()])
    X_meanstd = X_meanstd.reshape([16,6])

    X_hist = []
    for h in nHist:
        X_hist = np.concatenate([X_hist, h])
    X_hist = X_hist.reshape([16, 512])

    # Lets get some confusion matrices
    Y_mean = sp.spatial.distance.pdist(X_mean, 'euclidean')
    Y_meanstd = sp.spatial.distance.pdist(X_meanstd, 'euclidean')
    Y_hist = sp.spatial.distance.pdist(X_hist, 'euclidean')

    Y_mean = sp.spatial.distance.squareform(Y_mean)
    Y_meanstd = sp.spatial.distance.squareform(Y_meanstd)
    Y_hist = sp.spatial.distance.squareform(Y_hist)
    

    plt.figure()
    
    plt.subplot(131)
    plt.imshow(Y_mean / Y_mean.sum(), interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Mean')

    plt.subplot(132)
    plt.imshow(Y_meanstd / Y_meanstd.sum(), interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('MeanStd')

    plt.subplot(133)
    plt.imshow(Y_hist / Y_hist.sum(), interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Hist')
    
    
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

