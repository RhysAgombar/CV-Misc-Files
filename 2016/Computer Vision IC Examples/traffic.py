# -*- coding: utf-8 -*-
import cv2
import numpy as np
import scipy as sp
from scipy import stats
import argparse
import matplotlib.pyplot as plt

def process_vid(vidfile):
   cap = cv2.VideoCapture(vidfile)

   # BGR
   R = 2
   B = 0
   G = 1
   
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
       road_model[B,i], road_model[G,i], road_model[R,i], test = cv2.mean(road)
       bkgnd_model[B,i], bkgnd_model[G,i], bkgnd_model[R,i], test = cv2.mean(frame_640x480)
       #road_model[​B,i], road_model[G,i], road_model[R,i], _​ = cv2.mean(road)
       #bkgnd_model[​B,i], bkgnd_model[G,i], bkgnd_model[R,i], _​ = cv2.mean(frame_640x480)

       i += 1
       
       if cv2.waitKey(1) & 0xFF == ord('q'):
           break

   cap.release()

   road_model.shape
   
   # Lets try to plot these values.
   fig1 = plt.figure();
   ax = plt.subplot(1,1,1)

   ax.plot(road_model[R,:],'r', label='Road red')
   ax.plot(road_model[G,:],'g', label='Road green')
   ax.plot(road_model[B,:],'b', label='Road blue')

   ax.plot(bkgnd_model[R,:],'r.', label='Background red')
   ax.plot(bkgnd_model[G,:],'g.', label='Background green')
   ax.plot(bkgnd_model[B,:],'b.', label='Background blue')
   handles, labels = ax.get_legend_handles_labels()
   ax.legend(handles, labels)
   
   plt.show()
   
   # So this is what our system has learnt about the road model
   r_model_mean = np.mean(road_model, 1)
   b_model_mean = np.mean(bkgnd_model, 1)
   print 'road_model', r_model_mean
   print 'bkgnd_model', b_model_mean 
   
   # Now lets label the image
   label_img = np.zeros([480, 640, 3], dtype='float32')
   rr = label_img[:,:,R]
   bb = label_img[:,:,B]
   gg = label_img[:,:,G]
   
   mean_img = np.zeros([480, 640, 3])
   mean_img[:,:,R] = r_model_mean[R]
   mean_img[:,:,B] = r_model_mean[B]
   mean_img[:,:,G] = r_model_mean[G]
   
   cap = cv2.VideoCapture(vidfile)
   while(cap.isOpened()):
       ret, frame = cap.read()

       frame_640x480 = cv2.resize(frame, (640, 480))
       cv2.imshow('frame', frame_640x480)

       # Lets check for roads
       diff = np.sum((frame_640x480 - mean_img)**2, 2)
       diff_f = diff.astype('float32')
       diff_f /= np.max(diff)
       # r = sp.stats.threshold(diff_f, threshmin=0.1, newval=0.0)
       # r = sp.stats.threshold(r, threshmax=0.1, newval=1.0)

       print diff_f.shape
       print label_img.shape
       
       gg[np.where(diff_f < 0.1)] = 1.0
       rr[np.where(diff_f >= 0.1)] = 1.0
       
       cv2.imshow('diff', label_img)

       gg[:] = 0
       bb[:] = 0
       rr[:] = 0
       
       if cv2.waitKey(1) & 0xFF == ord('q'):
           break

   cap.release()
   cv2.destroyAllWindows()


if __name__ == '__main__':
   parser = argparse.ArgumentParser(description='CSCI 4220U Lab 2.')
   parser.add_argument('vidfile', help='Video file')
   args = parser.parse_args()

   process_vid(args.vidfile)