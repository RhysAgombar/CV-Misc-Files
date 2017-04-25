import cv2
import numpy as np
import scipy as sp
from scipy.cluster.vq import kmeans,vq
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

N = 100
mean1 = [0,0,0]
cov1 = [[1,0,0],[0,1,0],[0,0,1]] 
X1 = np.random.multivariate_normal(mean1, cov1, N)
#print X1.shape

mean2 = [2,2,0]
cov2 = [[1,0,0],[0,1,0],[0,0,1]] 
X2 = np.random.multivariate_normal(mean2, cov2, N)
#print X2.shape

X = np.vstack((X1, X2))

K = 4
centers, _ = kmeans(X, K)
#classified_points, _ = vq(X,centers) <- this would classify the points

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(centers[:,0], centers[:,1], centers[:,2])
ax.set_xlabel('Hue')
ax.set_ylabel('Saturation')
ax.set_zlabel('Value')
plt.show()