

from numpy import *
################## test svm #####################
## step 1: load data
print ("step 1: load data of train set...")
file_train = open(r'F:\something\SVM\test\N.txt')
lines_train = file_train.readlines()
nrows_train = len(lines_train)
ncols_train = len(lines_train[0].strip().split('\t'))-2  #exclude first two
traindata = zeros((nrows_train,ncols_train))
labels_train = []
i_train=0
for line in lines_train:
    linedata=[]
    lineArr = line.strip().split('\t')
    labels_train.append(float(lineArr[1]))
    for j in range(2,len(lineArr),1):
        if len(lineArr[j].strip())<1:
            lineArr[j]=0                 #replace null with 0
        linedata.append([float(lineArr[j])])
    #print linedata
    l = matrix(linedata).reshape(1,len(linedata))
    traindata[i_train,:]= l
    i_train += 1
labels_train = mat(labels_train).T
print traindata
#print labels
print ("step 1: load data of test set...")
file_test = open(r'F:\something\SVM\test\X.txt')
lines_test = file_test.readlines()
nrows_test = len(lines_test)
ncols_test = len(lines_test[0].strip().split('\t'))-2  #exclude first two
testdata = zeros((nrows_test,ncols_test))
labels_test = []
i_test=0
for line in lines_test:
    linedata=[]
    lineArr = line.strip().split('\t')
    labels_test.append(float(lineArr[1]))
    for j in range(2,len(lineArr),1):
        if len(lineArr[j].strip())<1:
            lineArr[j]=0                 #replace null with 0
        linedata.append([float(lineArr[j])])
    #print linedata
    l = matrix(linedata).reshape(1,len(linedata))
    testdata[i_train,:]= l
    i_test += 1
labels_test = mat(labels_test).T
print testdata
print("load data finish")
