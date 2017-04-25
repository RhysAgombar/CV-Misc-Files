# sampling.py
import argparse
import cv2
import numpy as np

def load_image(imgfile):
    print 'Opening ', imgfile
    img = cv2.imread(imgfile)

    cv2.imshow('Input image',img)

    b = img[::2,::2,:].copy()
    cv2.imshow('Input image',b)

    print 'Press any key to proceed'
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CSCI 4220U Lab 2.')
    parser.add_argument('--use-plotlib', action='store_true', help='If specified uses matplotlib for displaying images.')
    parser.add_argument('imgfile', help='Image file')
    args = parser.parse_args()

    load_image(args.imgfile)