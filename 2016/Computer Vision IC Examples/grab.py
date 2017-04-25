import numpy as np
import cv2
from matplotlib import pyplot as plt

sbox = None
cur_mouse = None
ebox = None
def on_mouse(event, x, y, flags, params):
    global sbox
    global cur_mouse
    global ebox
    
    if event == cv2.EVENT_LBUTTONDOWN:
        print 'Start Mouse Position: '+str(x)+', '+str(y)
        sbox = (x, y)
        ebox = None
    elif event == cv2.EVENT_MOUSEMOVE:
        # print 'Current Mouse Position: '+str(x)+', '+str(y)
        cur_mouse = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        print 'End Mouse Position: '+str(x)+', '+str(y)
        ebox = (x, y)


img = cv2.imread('messi.jpg')
cv2.namedWindow('Image')
cv2.setMouseCallback('Image', on_mouse, 0)

mask = np.zeros(img.shape[:2],np.uint8)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')

bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)


while(1):
    tmp = img.copy();
    if sbox and not ebox: 
        cv2.rectangle(tmp, sbox, cur_mouse, color=(0,255,0), thickness=1, lineType=8, shift=0) 
    elif sbox and ebox:
        cv2.rectangle(img, sbox, ebox, color=(255,0,0), thickness=1, lineType=8, shift=0)
        
    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
        
    cv2.imshow('Image', tmp)
    if cv2.waitKey(1) == 27:
        break;