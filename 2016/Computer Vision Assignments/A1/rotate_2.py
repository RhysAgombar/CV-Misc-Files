import argparse
import cv2
import numpy as np
import scipy as sp
import math

## NOTE: this does not work, I have only uploaded this file so you can see what I tried to accomplish.

def process_img(imgfile, angle):

    img = cv2.imread(imgfile)

    x, y, _ = img.shape

    hyp = int(math.sqrt(pow(img.shape[0], 2) + pow(img.shape[1], 2)))
    xadj = (hyp - x)/2
    yadj = (hyp - y)/2
    center = (hyp/2, hyp/2)

    rImg = np.zeros((hyp, hyp, 3), dtype='uint8')

    rImg[xadj:(xadj + x), yadj:(yadj + y), :] = img
    rotation = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    R = cv2.getRotationMatrix2D(center, angle, 1.0)
    R2 = [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]] 

    nImg = np.zeros((hyp, hyp, 3), dtype='uint8')

    for i in range((center[0] - x/2),(center[0] + x/2)):
        for j in range((center[1] - y/2),(center[1] + y/2)):
            coord = np.array([[i - (center[0] - (x/2))],[j - (center[1] - (y/2))]])
            coord[0] = (R2[0][0] * coord[0] +  R2[1][0] * coord[1])
            coord[1] = (R2[0][1] * coord[1] +  R2[1][1] * coord[1])
    
            coord = ((coord[0] + (center[0] - x/2)),(coord[0] + (center[1] - y/2)))
            nx = int(coord[0])
            ny = int(coord[1])

            nImg[nx][ny][:] = rImg[i][j][:] ## something's wrong

    cv2.imshow('test', nImg)

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

    process_img(args.imgfile, float(args.angle))
