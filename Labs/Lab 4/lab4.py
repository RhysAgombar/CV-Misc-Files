import argparse
import cv2
import random as r
import numpy as np
import scipy as sp
from scipy import signal
from matplotlib import pyplot as plt


A = np.array([[r.random()*10,r.random()*10,r.random()*10],[r.random()*10,r.random()*10,r.random()*10],[r.random()*10,r.random()*10,r.random()*10]])

startPoints = np.zeros([9,3])
endPoints = np.zeros([9,3])

for i in range (0,9):
    point = np.array([[r.random()*10,r.random()*10,r.random()*10]])
    startPoints[i][:] = point[:]
    ep = np.dot(point,A)
    endPoints[i][:] = ep[:]
    

startT = np.transpose(startPoints)

rs = np.dot(startT,endPoints)
ls = np.dot(startT, startPoints)
ls = np.linalg.inv(ls)

aEnd = np.dot(ls,rs)
out = A - aEnd
print A
print aEnd
print out
