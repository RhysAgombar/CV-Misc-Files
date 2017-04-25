# Assignment 2
import argparse
import cv2
import numpy as np
import os

breakpoint = 1185  # 1185: For some reason it explodes once you hit 1186. This stops that. 
                   # If your computer can handle more, adjust this value accordingly.
scale = 2 # I run out of RAM trying to do the computations at 1:1 scale. Downsizing it by a
          # factor of 2 makes execution possible.

def query(imgFile):
    print "Querying Database..."
    holder = np.fromfile(imgFile, dtype='uint8')  
    holder = np.reshape(holder[0:16384], (np.sqrt(16384),np.sqrt(16384)))

    cv2.imshow("Initial Image", holder)

    holder = holder[::scale,::scale].copy()
    holder = np.reshape(holder, (holder.shape[0]*holder.shape[1], 1))  

    comp = np.load("PCA-vec.npy")
    names = np.load("PCA-file.npy")
    karray = np.load("PCA-karray.npy")

    imgPCA = np.dot(karray, holder)
    
    imgSum = np.sum(imgPCA)
    compSum = np.zeros((comp.shape[1]))
    
    for i in range(comp.shape[1]):
        compSum[i] = np.sum(comp[:,i])
    
    
    pairs = [(np.abs(compSum[i] - imgSum), names[i]) for i in range(comp.shape[1])]
    pairs.sort(reverse=False, key=(lambda x: x[0]))

    k = 10

    for i in range(k):
        filepath = "rawdata/" + pairs[i][1]
        x = np.fromfile(filepath, dtype='uint8')
        x = np.reshape(x, (np.sqrt(16384),np.sqrt(16384)))
        cv2.imshow(str(i), x)
    
    
def do_pca(RawDataFolder):
    print "Beginning PCA Analysis. This may take a minute."
    first = True
    num = 1
    for filename in os.listdir(RawDataFolder):
        filepath = RawDataFolder + "/" + filename
        f = open(filepath,'rb')
        if (first == True):
            holder = np.fromfile(f, dtype='uint8')
            holder = np.reshape(holder[0:16384], (np.sqrt(16384),np.sqrt(16384)))[::scale,::scale].copy()
            holder = np.reshape(holder, (1, holder.shape[0]*holder.shape[1]))
            x = holder
            first = False
        else:
            holder = np.fromfile(f, dtype='uint8')
            holder = np.reshape(holder[0:16384], (np.sqrt(16384),np.sqrt(16384)))[::scale,::scale].copy()
            holder = np.reshape(holder, (1, holder.shape[0]*holder.shape[1]))
            x = np.vstack((x,holder))

            if (num == breakpoint): # 1185: For some reason it explodes once you hit 1186. This stops that until I find a solution.
                break
            num = num + 1

    K,N = x.shape
#
# COMMENT OUT TO SAVE RUN TIME DURING TESTS.
#    
    means = np.zeros(N)    
    Z = np.zeros((K,N))  
    
    for i in range (0, N):
        means[i] = (np.mean(x[:,i])/K)
        Z[:, i] = x[:, i] - means[i]
    
    Zt = np.dot(Z.T,Z)
    
    eigVals, eigVecs = np.linalg.eig(Zt)

    eigVals = eigVals.astype('float64')
    eigVecs = eigVecs.astype('float64')
    np.save("PCA-eigVals",eigVals)
    np.save("PCA-eigVecs",eigVecs)
#
    #eigVals = np.load("PCA-eigVals.npy")
    #eigVecs = np.load("PCA-eigVecs.npy")

    pairs = [(np.abs(eigVals[i]), eigVecs[:,i]) for i in range(len(eigVals))]
    pairs.sort(reverse=True, key=(lambda x: x[0]))
   
    topK = N/10
   
    karray = np.zeros([topK,N])
    for i in range(topK):
        karray[i][:] = pairs[i][1]
    
    np.save("PCA-karray",karray)

    first = True
    num = 1
    for filename in os.listdir("rawdata"):
        filepath = RawDataFolder + "/" + filename
        f = open(filepath,'rb')
        if (first == True):
            holder = np.fromfile(f, dtype='uint8')
            holder = np.reshape(holder[0:16384], (np.sqrt(16384),np.sqrt(16384)))[::scale,::scale].copy()
            holder = np.reshape(holder, (holder.shape[0]*holder.shape[1], 1))
            img = holder
            comp = np.dot(karray, img)
            names = filename
            first = False
        else:
            holder = np.fromfile(f, dtype='uint8')
            holder = np.reshape(holder[0:16384], (np.sqrt(16384),np.sqrt(16384)))[::scale,::scale].copy()
            holder = np.reshape(holder, (holder.shape[0]*holder.shape[1], 1))
            img = holder
            comp = np.hstack((comp,np.dot(karray, img)))
            names = np.hstack((names,filename))
            if (num == breakpoint):
                break
            num = num + 1

    np.save("PCA-vec",comp)
    np.save("PCA-file",names)
    print "PCA Analysis Complete."
    
def add_face(imgFile):
    print "Adding face to database..."
    holder = np.fromfile(imgFile, dtype='uint8')    
    
    print "Please enter the raw images folder location: "
    filepath = raw_input() 
    holder.tofile(filepath + '\\' + imgFile, sep="")
    print "Face added to Raw Images folder. Recomputing PCA..."
    
    do_pca(filepath)
    
    print "Image has been successfully added, and the PCA values have been updated."
    
def mid_PCA(x):
    K,N = x.shape
    
    means = np.zeros(N)    
    Z = np.zeros((K,N))  
    
    for i in range (0, N):
        means[i] = (np.mean(x[:,i])/K)
        Z[:, i] = x[:, i] - means[i]
    
    Zt = np.dot(Z.T,Z)
    
    eigVals, eigVecs = np.linalg.eig(Zt)

    eigVals = eigVals.astype('float64')
    eigVecs = eigVecs.astype('float64')
    np.save("PCA-eigVals-mid",eigVals)
    np.save("PCA-eigVecs-mid",eigVecs)

    pairs = [(np.abs(eigVals[i]), eigVecs[:,i]) for i in range(len(eigVals))]
    pairs.sort(reverse=True, key=(lambda x: x[0]))
   
    topK = N/10
   
    karray = np.zeros([topK,N])
    for i in range(topK):
        karray[i][:] = pairs[i][1]
    
    np.save("PCA-karray-mid",karray)
    
def con_mat(RawDataFolder):
    print "Computing Confusion Matrix. This may take a while..."
    matrix = np.zeros([breakpoint/3,breakpoint/3,1])
    num = 0
    j = 0
    first = True
        
    for filename in os.listdir(RawDataFolder):
        filepath = RawDataFolder + "/" + filename
        f = open(filepath,'rb')
        if (num < (breakpoint - breakpoint/3)):
            if (first == True):
                holder = np.fromfile(f, dtype='uint8')
                holder = np.reshape(holder[0:16384], (np.sqrt(16384),np.sqrt(16384)))[::scale,::scale].copy()
                holder = np.reshape(holder, (1, holder.shape[0]*holder.shape[1]))
                x = holder
                first = False
            else:
                holder = np.fromfile(f, dtype='uint8')
                holder = np.reshape(holder[0:16384], (np.sqrt(16384),np.sqrt(16384)))[::scale,::scale].copy()
                holder = np.reshape(holder, (1, holder.shape[0]*holder.shape[1]))
                x = np.vstack((x,holder))
        elif (num == (breakpoint - breakpoint/3)):
            mid_PCA(x)
            karray = np.load("PCA-karray-mid.npy")
            first = True
        else:
            if (first == True):
                holder = np.fromfile(f, dtype='uint8')
                holder = np.reshape(holder[0:16384], (np.sqrt(16384),np.sqrt(16384)))[::scale,::scale].copy()
                holder = np.reshape(holder, (1, holder.shape[0]*holder.shape[1]))
                x_sub = holder
                first = False
            else:
                holder = np.fromfile(f, dtype='uint8')
                holder = np.reshape(holder[0:16384], (np.sqrt(16384),np.sqrt(16384)))[::scale,::scale].copy()
                holder = np.reshape(holder, (1, holder.shape[0]*holder.shape[1]))
                x_sub = np.vstack((x_sub,holder))
            
        if (num == breakpoint):
            break
        num = num + 1

    rows = x_sub.shape[0]

    #for i in range (rows):
    #    img = np.dot(karray, x_sub[i][:].T)
    #    isum = np.sum(img)
    #    for j in range(rows):
    #        comp = np.dot(karray, x_sub[j][:].T)
    #        csum = np.sum(comp)
    #        matrix[i][j] = np.abs(isum - csum)

    #np.save("matrix",matrix)
    matrix = np.load("matrix.npy")
    cv2.imshow("Confusion Matrix", matrix)
    
    matrix = matrix/np.amax(matrix)
    cv2.imshow("Confusion Matrix (Normalized)", matrix)
    
    matrix = matrix * 255
    cv2.imshow("Confusion Matrix (Adjusted)", matrix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()      
    parser.add_argument("--do_pca")
    parser.add_argument("--add_face")
    parser.add_argument("--query")
    parser.add_argument("--confusion_matrix")
    args = parser.parse_args()
    
    if args.do_pca:
        do_pca(args.do_pca)
    if args.query:
        query(args.query)
    if args.add_face:
        add_face(args.add_face)
    if args.confusion_matrix:
        con_mat(args.confusion_matrix)