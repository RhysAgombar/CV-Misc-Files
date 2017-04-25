import argparse
import cv2
import numpy as np

def blend(alpha, img1, img2):
    imgL1 = cv2.imread(img1) 
    imgL2 = cv2.imread(img2) 
    nImg1 = alpha * imgL1
    nImg2 = (1 - alpha) * imgL2
    
    X1,Y1,Z1 = nImg1.shape
    X2,Y2,Z2 = nImg2.shape
    
    imgOut = np.zeros((max(X1,X2),max(Y1,Y2),max(Z1,Z2)))
    imgOut[0:X1,0:Y1,0:Z1] += nImg1[:,:,:]
    imgOut[0:X2,0:Y2,0:Z2] += nImg2[:,:,:]
    
    cv2.imshow('Blended Image',cv2.convertScaleAbs(imgOut))
    cv2.imshow('Cropped to First Image',cv2.convertScaleAbs(imgOut[0:X1,0:Y1,0:Z1]))
    cv2.imshow('Cropped to Smallest Image',cv2.convertScaleAbs(imgOut[0:min(X1,X2),0:min(Y1,Y2),0:min(Z1,Z2)]))
    
    print 'Press any key to proceed'   
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CSCI 4220U Lab 2.')
    parser.add_argument('--use-plotlib', action='store_true', help='If specified uses matplotlib for displaying images.')
    parser.add_argument('alpha', help='Alpha')
    parser.add_argument('imgfile', help='Image file')
    parser.add_argument('imgfile2', help='Image file')
    args = parser.parse_args()
    
    blend(float(args.alpha), args.imgfile, args.imgfile2)