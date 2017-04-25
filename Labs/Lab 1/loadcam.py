from cv2.cv import *
from cv2 import *
import sys

port = 0
cam = VideoCapture(port)
nixd, img = cam.read()
imshow("Camimg",img)

WaitKey(0)
del(cam)