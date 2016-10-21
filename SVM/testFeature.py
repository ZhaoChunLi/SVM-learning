
#....choose features of the data
#....test the good features

import SVM
import testexcel

def getmax_pos (wfactors_abs,wfactors,num):
    length = wfactors.shape[1]
    pos = []
    pos_re=[]
    valueOFw = []
    posnum = 0
    for i in range(length):
        pos.append(wfactors_abs[0,i])
    while (posnum<num):
        maxnum=max(pos)
        index = pos.index(maxnum)
        valueOFw.append(wfactors[0,index])
        pos_re.append(index+1)
        pos[pos.index(maxnum)] = 0
        posnum = posnum + 1
    return pos_re,valueOFw

def savefeature(data_path,feature_path,featurepos):
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

if __name__== '__main__':
    print "loading data..."
    traindataF, trainlabelsF = testexcel.loaddata_normalize(r'F:\something\SVM\test\pythontest\NFeat.xls', 0)
    testdataF, testlabelsF = testexcel.loaddata_normalize(r'F:\something\SVM\test\pythontest\NFeat.xls', 1)
    # traindata, trainlabels = loaddata(r'F:\something\SVM\test\pythontest\GOOD\X.xlsx', 0)
    # testdata, testlabels = loaddata(r'F:\something\SVM\test\pythontest\GOOD\X.xlsx', 1)
    # traindata, trainlabels = loaddata(r'F:\something\SVM\test\pythontest\all\NX.xlsx', 0)
    # testdata, testlabels = loaddata(r'F:\something\SVM\test\pythontest\all\NX.xlsx', 1)
    print "trainning..."
    C = 0.6
    toler = 0.001
    maxIter = 50
    svmClassifierF = SVM.trainSVM(traindataF, trainlabelsF, C, toler, maxIter, kernelOption=('linear', 0))
    # svmClassifier = SVM.trainSVM(traindata, trainlabels, C, toler, maxIter, kernelOption=('rbf', 0.55))
    # N 8~9  X 0.3~0.55
    print "testing..."
    accuracyF, predictlabelsF = SVM.testSVM(svmClassifierF, testdataF, testlabelsF)
    print predictlabelsF
