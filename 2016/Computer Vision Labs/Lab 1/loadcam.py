from cv2 import *

port = 0
cam = VideoCapture(port)
nixd, img = cam.read()
imshow("Camimg",img)
del(cam)