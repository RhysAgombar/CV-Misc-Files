# -*- coding: utf-8 -*-
import cv2
import numpy as np
import argparse

def process_vid(vidfile):
   cap = cv2.VideoCapture(vidfile)

   # BGR
   _R = 2
   _B = 0
   _G = 1
   
   num_frames = 50
   road_model = np.zeros([3, num_frames])
   bkgnd_model = np.zeros([3, num_frames])

   i = 0
   while(cap.isOpened() and i < 50):
       ret, frame = cap.read()

       # If you want a grayscale image.
       # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
       # cv2.imshow('frame',gray)

       # Task 1: resize the input frames to 640x480
       frame_640x480 = cv2.resize(frame, (640, 480))
       # cv2.imshow('frame', frame_640x480)

       # Task 2: crop the image to right half
       h,w,channels =  frame_640x480.shape
       road = frame_640x480[300:, w/2:]
       cv2.imshow('frame', road)

       # Task 3: average color of frame_640x480 and road images.
       # print cv2.mean(road), cv2.mean(frame_640x480)

       # Task 4: learn a road and non-road (background) model
       print "i = ", i
       road_model[_B,i], road_model[_G,i], road_model[_R,i], _​ = cv2.mean(road)
       bkgnd_model[_B,i], bkgnd_model[_G,i], bkgnd_model[_R,i], _​ = cv2.mean(frame_640x480)

       i += 1
       
       if cv2.waitKey(1) & 0xFF == ord('q'):
           break

   cap.release()
       
   print 'road_model', np.mean(road_model, 1)
   print 'bkgnd_model', np.mean(bkgnd_model, 1)


   cap = cv2.VideoCapture(vidfile)
   while(cap.isOpened()):
       ret, frame = cap.read()
       
       frame_640x480 = cv2.resize(frame, (640, 480))
       cv2.imshow('frame', frame_640x480)
       
       if cv2.waitKey(1) & 0xFF == ord('q'):
           break

   cap.release()
   cv2.destroyAllWindows()


if __name__ == '__main__':
   parser = argparse.ArgumentParser(description='CSCI 4220U Lab 2.')
   parser.add_argument('vidfile', help='Video file')
   args = parser.parse_args()

   process_vid(args.vidfile)