import cv2
import numpy as np
import time
import math

def process_cam():
    cap = cv2.VideoCapture(0)

    while(True):
        time1 = time.time()

        # Capture frame-by-frame
        ret, frame = cap.read()
        ret2, frame2 = cap.read()
        red = frame[:,:,2] #<- gets only the red channel (0 is blue, 1 is green)

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        width, height = gray.shape     
        
        ratio = width/float(height)
        
        grayDown = cv2.resize(gray, (256, int(ratio*256))) ## Get aspect ratio working
        
        time2 = time.time()
        print '%s took %0.3f ms' % ('Image capture and grayscale conversion', (time2-time1)*1000.0)


        # Display the resulting frame
        cv2.imshow('Downsized',grayDown)
        #cv2.imshow('Red',red)
        cv2.imshow('Motion',gray-gray2)

        print 'Press q to stop'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    process_cam()