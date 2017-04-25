import numpy as np
from cv2 import *
import sys

video = VideoCapture("traffic-short.mp4")#sys.argv[1])

while(video.isOpened()):
    Unused, movie = video.read()
    imshow('Movie',movie)
    if waitKey(1) & 0xFF == ord('q'):
        break

video.release()
destroyAllWindows()