import argparse
import cv2
import numpy as np
import scipy as sp

img = cv2.imread("girl.jpg")
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow('standard', img)
cv2.imshow('grey', grayImg)