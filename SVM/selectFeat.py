
###select feature with kafang
import SVM
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
from numpy import *
import numpy as np
import xlrd
import SVM
import xlwt

def loadtrain(file_path):
    data = xlrd.open_workbook(file_path)
    train_table = data.sheets()[0]
    nrows1 = train_table.nrows
    ncols1 = train_table.ncols
    train = zeros((nrows1,ncols1-1))
    labels = []
    labels = mat(train_table.col_values(0)).T
    for x in range(nrows1):
        rows = train_table.row_values(x)
        rowmatrix = rows[1:]
        train[x,:] =  rowmatrix
    return train,labels
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
def savefeature(data_path,feature_path,featurepos,num):
    train_data, test_data = loaddata(data_path)
    feature = xlwt.Workbook()
    train_feature = feature.add_sheet("train",cell_overwrite_ok=True)
    test_feature = feature.add_sheet("test",cell_overwrite_ok=True )
    #numOFfeat = len(featurepos)
    numOFfeat = num
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

if __name__== '__main__':
    train,trainlabels= loadtrain(r'F:\something\SVM\test\pythontest\N.xlsx')
    y = []
    for i in range(len(trainlabels)):
        if trainlabels[i] == 1:
            y.append(True)
        else:
            y.append(False)
    y = np.array(y)
    train = np.array(train)
    transformer = SelectKBest(score_func=chi2, k=3)
    train_norm= MinMaxScaler().fit_transform(train)
    data = transformer.fit_transform(train_norm, y)
    sort_data = np.argsort(-transformer.scores_)
    pos = sort_data[0:899]
    data_path = r'F:\something\SVM\test\pythontest\N.xlsx'
    feature_path = r'F:\something\SVM\test\pythontest\NFeatsk.xls'
    savefeature(data_path, feature_path, pos,200)