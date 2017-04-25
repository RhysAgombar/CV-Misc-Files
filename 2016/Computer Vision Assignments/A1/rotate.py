import argparse
import cv2
import numpy as np
import scipy as sp
import math

def process_img(imgfile, angle):

    img = cv2.imread(imgfile)

    x, y, _ = img.shape

    hyp = int(math.sqrt(pow(img.shape[0], 2) + pow(img.shape[1], 2)))
    xadj = (hyp - x)/2
    yadj = (hyp - y)/2
    center = (hyp/2, hyp/2)

    rImg = np.zeros((hyp, hyp, 3), dtype='uint8')

    rImg[xadj:(xadj + x), yadj:(yadj + y), :] = img
    rotation = cv2.getRotationMatrix2D(center, float(angle), 1.0)
    R = cv2.getRotationMatrix2D(center, float(angle), 1.0)

    rotImg = cv2.warpAffine(rImg, rotation, (hyp,hyp),flags=cv2.INTER_LINEAR)

    x0, x1, x2, x3 = xadj, xadj + x, xadj, xadj + x
    y0, y1, y2, y3 = yadj, yadj, yadj + y, yadj + y

    points = np.zeros((3,4))
    points[0,0], points[0,1], points[0,2], points[0,3] = x0, x1, x2, x3
    points[1,0], points[1,1], points[1,2], points[1,3] = y0, y1, y2, y3
    points[2:] = 1

    imgFrame = np.dot(rotation, points)

    x = int(imgFrame[0,0])
    y = int(imgFrame[1,0])
    l,r,u,d = x,x,y,y

    for i in range(4):
        x = int(imgFrame[0,i])
        y = int(imgFrame[1,i])
        if (x < l): 
            l = x
        if (x > r): 
            r = x
        if (y < u): 
            u = y
        if (y > d): 
            d = y
    h = d - u
    w = r - l

    cropped = np.zeros((w, h, 3), dtype='uint8')
    cropped[:,:,:] = rotImg[l:(l+w), u:(u+h), :]

    cv2.imshow('rotated', cropped)

    print 'Press any key to proceed'
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CSCI 4220U Lab 2.')
    parser.add_argument('--use-plotlib', action='store_true', help='If specified uses matplotlib for displaying images.')
    parser.add_argument('imgfile', help='Image file')
    parser.add_argument('angle', help='Rotation Angle')
    args = parser.parse_args()

    process_img(args.imgfile, args.angle)
