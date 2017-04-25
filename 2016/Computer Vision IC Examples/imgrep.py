# imgrep.py
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

def makehist(img, name):
    raw = img.flatten()
    print '1. raw image =', raw.shape
    
    means = cv2.mean(img)
    print '2. mean =', means
    
    means, stds = cv2.meanStdDev(img)
    f = np.concatenate([means, stds]).flatten()
    print '3. mean and std =', f
    
    hist = cv2.calcHist(img, [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    print '4. image hist shape =', hist.shape
    hist = hist.flatten()
    print '   image hist feaure shape =', hist.shape
    center = np.linspace(0,512,512)
    
    plt.figure(figsize=(10,10))
    ax = plt.subplot(111)
    ax.bar(center, hist, align='center', width=.5)
    plt.title(name)
    
    plt.show()

def load_image(imgfile):
    print 'Opening ', imgfile
    img = cv2.imread("starry_night.jpg")
    img1 = cv2.imread("starry-night-vertical-flip.jpg")
    img2 = cv2.imread("starry-night-brigher.jpg")
    img3 = cv2.imread("starry-night-small.jpg")
    img4 = cv2.imread("test(2).jpg")

    print img

    cv2.imshow('0',img)
    cv2.imshow('1',img1)
    cv2.imshow('2',img2)
    cv2.imshow('3',img3)
    cv2.imshow('4',img4)  
    
    makehist(img,"starry_night.jpg")  
    makehist(img1,"starry-night-vertical-flip.jpg") 
    makehist(img2,"starry-night-brigher.jpg") 
    makehist(img3,"starry-night-small.jpg") 
    makehist(img4,"test(2).jpg") 

    print 'Press any key to proceed'   
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CSCI 4220U Lab 2.')
    parser.add_argument('imgfile', help='Image file')
    args = parser.parse_args()

    print args.imgfile

    load_image(args.imgfile)