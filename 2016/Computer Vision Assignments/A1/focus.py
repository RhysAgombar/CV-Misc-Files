# focus.py
import argparse
import cv2
import numpy as np
import time
import scipy as sp
from scipy.cluster.vq import kmeans,vq
from scipy import signal
#from matplotlib import pyplot as plt

def filter(img):
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    filterArray = np.array([[0,0.25,0],
                            [0.25,-1,0.25],
                            [0,0.25,0]])

    focus = signal.convolve2d(grayImg, filterArray, mode='same', boundary='fill')
    
    focus = sp.stats.threshold(focus, threshmin=20, newval=0.0)
    focus = sp.stats.threshold(focus, threshmax=1, newval=255.0)

    focus = cv2.convertScaleAbs(focus)
    
    return focus

def drawRect(img, x, y, random_points_list, classified_points, n):    
    minx, miny = 10000, 10000
    maxx, maxy = 0, 0
    
    for point, allocation in zip(random_points_list, classified_points):  
        if allocation == n:
            if point[0] < minx:
                minx = point[0]
            if point[0] > maxx:
                maxx = point[0]
            if point[1] < miny:
                miny = point[1]
            if point[1] > maxy:
                maxy = point[1]

    cv2.rectangle(img, (miny,minx), (maxy,maxx), (0,0,255), 2)
    
    return img
    
def rect_focus(oImg,img):
    x,y = img.shape

    points = []
    samples_list = np.array([0,0])
    for i in range (0,x):
        for j in range (0,y):
            if (img[i,j] > 0):
                points.append([i,j])             
    
    samples_list = np.array(points)

    random_points_list = np.array(samples_list, np.float32) 

    K = 4
    centers, _ = kmeans(random_points_list, K)
    classified_points, _ = vq(random_points_list,centers)

    for i in range(0,K):
        oImg = drawRect(oImg, x, y, random_points_list, classified_points, i)

    return oImg

def process_img(imgfile):
    start = time.time()
    print 'Opening ', imgfile
    img = cv2.imread(imgfile)  

    focus = filter(img)

    cv2.imshow('Input image',img)
    
    rect = rect_focus(img, focus)

    cv2.imshow('Focus Detection w/t Rectangles', rect)
    
    print 'Focus Analysis took {0}ms.'.format(int(round((time.time() - start) * 1000)))

    print 'Press any key to proceed'   
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CSCI 4220U Lab 2.')
    parser.add_argument('--use-plotlib', action='store_true', help='If specified uses matplotlib for displaying images.')
    parser.add_argument('imgfile', help='Image file')
    args = parser.parse_args()
    
    process_img(args.imgfile)
