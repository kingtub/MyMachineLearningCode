from numpy import *
from sklearn.ensemble import AdaBoostClassifier


def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().strip().split('\t'))
    dataMat = []
    labelMat = []
    for line in open(fileName).readlines():
        lineArr = []
        arr = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(arr[i]))
        labelMat.append(float(arr[-1]))
        dataMat.append(lineArr)
    return dataMat, labelMat


def testing2():
    trainingDataMat, trainingClassLabels = loadDataSet('horseColicTraining2.txt')
    testDataMat, testClassLabels = loadDataSet('horseColicTest2.txt')
    bdt = AdaBoostClassifier()
    bdt.fit(trainingDataMat, trainingClassLabels)
    predicted = bdt.predict(testDataMat)
    m = len(testClassLabels)
    for i in range(m):
        print('p{0}={1}, r{2}={3}'.format(i, predicted[i], i, testClassLabels[i]))
    testLabelMat = mat(testClassLabels, dtype=float).T
    incorrect = zeros((m, 1))
    incorrect[testLabelMat != mat(predicted).T] = 1
    print('Error rate is ', float(incorrect.sum())/m) #   0.22388059701492538


testing2()