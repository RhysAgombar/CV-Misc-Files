#from mnist import MNIST
import cPickle, gzip
import numpy as np
import cv2
import numpy as np

f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

test = np.reshape(train_set[0][0], (28,28))

'''
cv2.imshow(str(train_set[1][0]),test)
print train_set[0][0].shape
'''

size = train_set[0].shape[0]


knn = cv2.KNearest()
knn.train(train_set[0][0:size],train_set[1][0:size])
ret,result,neighbours,dist = knn.find_nearest(test_set[0][0:size],k=1)


reslst = []
for i in range(0, result.shape[0]):  
    reslst.append(abs(result[i][0] - test_set[1][i]))

wrong = 0
for i in range(0, len(reslst)): 
    if (reslst[i] > 0.0):
        wrong += 1 

print "Accuracy: " + str(1.0 - (wrong / float(size)))

