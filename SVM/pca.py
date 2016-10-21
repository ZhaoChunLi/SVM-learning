
from numpy import *
import numpy as np
import SVM
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def pca(dataMat, topNfeat=9999999):
    meanVals = mean(dataMat, axis=0)   #caculate mean of each col
    meanRemoved = dataMat - meanVals   #remove mean
    covMat = cov(meanRemoved, rowvar=0)
    eigVals,eigVects = linalg.eig(mat(covMat))
    eigValInd = argsort(eigVals)                      #index, sort goes smallest to largest
    #print eigValInd
    eigValInd = eigValInd[:-(topNfeat+1):-1]  #cut off unwanted dimensions
    redEigVects = eigVects[:,eigValInd]       #reorganize eig vects largest to smallest
    lowDDataMat = meanRemoved * redEigVects   #transform data into new dimensions
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat

def plot2(dataSet1,y):
    dataArr1 = array(dataSet1)
    n = shape(dataArr1)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    j=0
    for i in range(n):
        xcord1.append(real(dataArr1[i,0]))
        ycord1.append(real(dataArr1[i,1]))
    traindata = zeros((len(xcord1),2))
    traindata[:,0]=xcord1
    traindata[:,1]=ycord1
    C = 0.6
    toler = 0.001
    maxIter = 50
    min_x,max_x,min_y,max_y = plotline(traindata,y,C, toler, maxIter)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(len(y)):
        if y[i]==1:
            ax.scatter(xcord1[i], ycord1[i], s=30, c='red', marker='s')
            #plt.plot(xcord1[i], ycord1[i], 'or')
        else:
            ax.scatter(xcord1[i], ycord1[i], s=30, c='blue', marker='s')
            #plt.plot(xcord1[i], ycord1[i], 'ob')
    plt.plot([min_x, max_x], [min_y,max_y], '-g')
    plt.xlabel('X1'); plt.ylabel('X2')
    plt.show()

def plotline(traindata,trainlabels ,C, toler, maxIter):
    svmClassifier = SVM.trainSVM(traindata, trainlabels, C, toler, maxIter, kernelOption=('linear', 0))
    w = SVM.calcWfactors(traindata, trainlabels, svmClassifier.alphas)
    print "hi traindata"
    print w
    min_x = min(traindata[:, 0])
    max_x = max(traindata[:, 0])
    y_min_x = float(-svmClassifier.b - w[0,0] * min_x) / w[0,1]
    y_max_x = float(-svmClassifier.b - w[0,0] * max_x) / w[0,1]
    return  min_x,max_x,y_min_x,y_max_x

def plot3(dataSet1,y):
    dataArr1 = array(dataSet1)
    n = shape(dataArr1)[0]
    xcord1 = []
    ycord1 = []
    zcord1 = []
    j = 0
    for i in range(n):
        xcord1.append(dataArr1[i, 0])
        ycord1.append(dataArr1[i, 1])
        zcord1.append(dataArr1[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    for i in range(n):
        if (y[i]==1):
            ax.scatter(xcord1[i], ycord1[i],zcord1[i], c='y')
        else:
            ax.scatter(xcord1[i], ycord1[i],zcord1[i], c='r')
    ax.set_zlabel('Z')
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.show()

if __name__=='__main__':
     #mata=loadDataSet('/Users/hakuri/Desktop/testSet.txt')
     #a,b= pca(mata, 2)
     #plotBestFit(a,b)
     data=np.array([[1, -1], [1, 1]])
     w, v = linalg.eig(data)
     print v
     print v*v.T
     print data*v*v.T
