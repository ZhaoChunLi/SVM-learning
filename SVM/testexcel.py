 #format:excel
#two files separate
#first replace null with 0
from numpy import *
import numpy as np
import xlrd
import SVM
import xlwt
import pca
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import matplotlib.pyplot as plt

#import showROC

def loaddata_normalize(file_path,sheetnum):
    data = xlrd.open_workbook(file_path)
    table = data.sheets()[sheetnum]
    nrows = table.nrows 
    ncols = table.ncols
    labels = []
    labels = mat(table.col_values(0)).T
    datamatrix=zeros((nrows,ncols-1))
    for x in range(nrows): 
        rows = table.row_values(x)
        rowmatrix = rows[1:]
        datamatrix[x,:] =  rowmatrix
    data_matrix = normalize(datamatrix)
    return data_matrix,labels

def loaddata(file_path):
    data = xlrd.open_workbook(file_path)
    train_table = data.sheets()[0]
    test_table = data.sheets()[1]
    nrows1 = train_table.nrows
    ncols1 = train_table.ncols
    nrows2 = test_table.nrows
    ncols2 = test_table.ncols
    train = zeros((nrows1,ncols1))
    test = zeros((nrows2,ncols2))
    for x in range(nrows1):
        rows = train_table.row_values(x)
        rowmatrix = rows[:]
        train[x,:] =  rowmatrix
    for x in range(nrows2):
        rows = test_table.row_values(x)
        rowmatrix = rows[:]
        test[x,:] =  rowmatrix
    return train,test

def normalize(matrix):
    length = len(matrix[1])
    for i in range(length):
        cols = matrix[:,i]
        minVals = min(cols)
        maxVals = max(cols)
        ranges = maxVals - minVals
        if ranges == 0:
            continue
        r =mat(cols)
        b = r - minVals
        normcols = b/ranges
        matrix[:,i]= normcols
    return matrix

def getsortedpos (wfactors_abs,wfactors):
    length = wfactors.shape[1]
    pos = []
    pos_re=[]
    valueOFw = []
    for i in range(length):
        pos.append(wfactors_abs[0,i])
    for j in range(length):
        maxnum=max(pos)
        index = pos.index(maxnum)
        valueOFw.append(wfactors[0,index])
        pos_re.append(index+1)
        pos[pos.index(maxnum)] = 0
    return pos_re,valueOFw

def savefeature(data_path,featurepath,featurepos):
    train_data, test_data = loaddata(data_path)
    feature = xlwt.Workbook()
    train_feature = feature.add_sheet("train",cell_overwrite_ok=True)
    test_feature = feature.add_sheet("test",cell_overwrite_ok=True )
    numOFfeat = len(featurepos)
    numOFtrain = train_data.shape[0]
    numOFtest = test_data.shape[0]
    #write labels
    for j in range(numOFtrain):
        label = train_data[j, 0]
        train_feature.write(j,0,train_data[j,0])
    #write features
    for i in range(numOFfeat):
        pos = featurepos[i]
        for j in range(numOFtrain):
            train_feature.write(j,i+1,train_data[j,pos])

    for j in range(numOFtest):
        label = test_data[j, 0]
        test_feature.write(j,0,test_data[j,0])
    for i in range(numOFfeat):
        pos = featurepos[i]
        for j in range(numOFtest):
            test_feature.write(j,i+1,test_data[j,pos])
    feature.save(feature_path)

def saveFeatRFE(train_x,y,alphas,k,w2Rank):
    length=len(w2Rank)
    print "paozhena"
    print length
    train_data=train_x
    num = len(alphas)
    numFeat = train_x.shape[1]
    print numFeat
    if length<numFeat:
        w2=0
        minmember=0
        w2_elimi=0
        e=0
        w2score=[]
        for i in range(num):
            kernel_row1 =np.array( k[i])
            for j in range(num):
                 w2 = w2 + alphas[i] * alphas[j] * y[i]*y[j]*kernel_row1[0][j]
        for r in range(numFeat):
            print r
            train_data[:,r]=0
            kk = SVM.calcKernelMatrix(mat(train_data), kernelOption = ('linear', 0))
            for ii in range(num):
                kernel_row2 = np.array( k[ii] )
                for jj in range(num):
                    w2_elimi = w2_elimi + alphas[ii] * alphas[jj] * y[ii] * y[jj] * kernel_row2[0][jj]
            e=abs(w2-w2_elimi)
            w2score.append(e)
        minmember = min(w2score)
        ind = w2score.index(minmember)
        w2Rank.insert(0,ind)
        train_x[:,ind]=0
        k = SVM.calcKernelMatrix(mat(train_x),  kernelOption = ('linear', 0))
        saveFeatRFE(train_x, y, alphas, k, w2Rank)
    else:
        return w2Rank
def getfeature(traindata,testdata,featurepos,numOFfeat):
    numOFtrain = traindata.shape[0]
    numOFtest = testdata.shape[0]
    trainFeat = zeros((numOFtrain, numOFfeat))
    testFeat = zeros((numOFtest, numOFfeat))
    for i in range(numOFfeat):
        pos = featurepos[i]
        trainFeat[:,i]=traindata[:,pos-1]
    for i in range(numOFfeat):
        pos = featurepos[i]
        testFeat[:,i]=testdata[:,pos-1]
    return trainFeat,testFeat
def getlabelsOFfeat(traindata, trainlabels, testdata,testlabels,alphas,numOFfeat):
    wfactors = SVM.calcWfactors(traindata, trainlabels, alphas)
    wfactors_abs = abs(wfactors)
    pos = []
    pos,valueOFw = getsortedpos(wfactors_abs,wfactors)
    trainFeat,testFeat = getfeature(traindata,testdata,pos,numOFfeat)
    print "training..."
    C = 0.6
    toler = 0.001
    maxIter = 50
    svmClassifier = SVM.trainSVM(trainFeat, trainlabels, C, toler, maxIter, kernelOption=('linear', 0))
    print "testing..."
    accuracy, predictlabels = SVM.testSVM(svmClassifier, testFeat, testlabels)
    return predictlabels

if __name__ == '__main__':
    print "loading data..."
    traindata,trainlabels = loaddata_normalize(r'F:\something\SVM\test\pythontest\N.xlsx',0)
    testdata,testlabels = loaddata_normalize(r'F:\something\SVM\test\pythontest\N.xlsx',1)
    print "trainning..."
    C = 0.6
    toler = 0.001 
    maxIter = 50   
    svmClassifier = SVM.trainSVM(traindata, trainlabels, C, toler, maxIter, kernelOption = ('linear', 0))
    #svmClassifier = SVM.trainSVM(traindata, trainlabels, C, toler, maxIter, kernelOption=('rbf', 0.55))
    #N 8~9  X 0.3~0.55
    print "testing..."
    accuracy,predictlabels = SVM.testSVM(svmClassifier, testdata, testlabels)
    print predictlabels
    print "wfactors..."
    numOFfeat = 500
    predictl = getlabelsOFfeat(traindata, trainlabels,testdata,testlabels,svmClassifier.alphas,numOFfeat)
    print predictl
    #print "svm alpha factors and b..."
    # print svmClassifier.alphas.T
    # print svmClassifier.b
    lowDDataMat, reconMat=pca.pca(traindata,2)
    pca.plot2(lowDDataMat,trainlabels)
    #pca.plotline(lowDDataMat,trainlabels, C, toler, maxIter)
    #lowDDataMat2, reconMat2 = pca.pca(traindata,3)
    #pca.plot3(lowDDataMat2,trainlabels)

