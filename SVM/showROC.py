#show ROC for svm results

from numpy import *
import pylab as pl

#############将全部的数据分成几个部分来实现，将最终的结果读出，依次求解
#############需要较多的数据来实现
########最好能将结果写进一个文件，格式为真正数，负正数。
def show(testlabels,predictlabels):
    tp = 0
    fp = 0
    pos = 0
    neg = 0
    y = 0
    xy_arr = []
    for i in range(len(testlabels)):
        if(predictlabels[i]==1 and testlabels[i] == 1):
            pos = pos + 1
        if (predictlabels[i] == -1 and testlabels[i] == 1):
            neg = neg + 1
    print pos
    print neg
    for j in range(len(testlabels)):
        if ((predictlabels[j] == 1) and (testlabels[j] == 1)):
            tp = tp + 1
        if (predictlabels[j] == -1 and testlabels[j] == 1):
            fp = fp + 1
        xy_arr.append([float(fp)/ neg,float(tp)/pos])
    for j in range(len(testlabels)): #calclate area under ROC
        auc = 0.
        prev_x = 0
    for x, y in xy_arr:
        if x != prev_x:
            auc += (x - prev_x) * y
            prev_x = x
    print xy_arr
    draw(xy_arr,auc)

def draw(xy_arr,auc):
    x = [_v[0] for _v in xy_arr]
    y = [_v[1] for _v in xy_arr]
    pl.title("ROC curve of %s (AUC = %.4f)" % ('svm', auc))
    pl.xlabel("False Positive Rate")
    pl.ylabel("True Positive Rate")
    pl.plot(x, y)  # use pylab to plot x and y
    pl.show()  # show the plot on the screen