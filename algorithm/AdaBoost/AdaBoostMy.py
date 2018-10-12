from numpy import *

def loadSimpData():
    datMat = matrix([[ 1. ,  2.1],
        [ 2. ,  1.1],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat,classLabels


def stumpClassify(dataMat, dim, threshold, inequal):
    m = dataMat.shape[0]
    retArr = ones((m, 1))
    if inequal == 'lt':
        retArr[dataMat[:, dim] <= threshold] = -1
    else:
        retArr[dataMat[:, dim] > threshold] = -1
    return retArr


def buildStump(dataArr, labels, D):
    dataMat = mat(dataArr, dtype=float)
    labelsMat = mat(labels, dtype=float).T
    m, n = dataMat.shape
    stepNums = 10
    stump = {}
    minError = inf
    predictedVals = None
    for i in range(n):
        rangeMin = dataMat[:, i].min()
        rangeMax = dataMat[:, i].max()
        stepValue = (rangeMax - rangeMin) / float(stepNums)
        for j in range(-1, stepNums + 1):
            threshold = rangeMin + j * stepValue
            for inequal in ['lt', 'gt']:
                retArr = stumpClassify(dataMat, i, threshold, inequal)
                error = mat(ones((m, 1)))
                error[retArr == labelsMat] = 0
                weightedError = D.T * error
                if float(weightedError) < minError:
                    minError = float(weightedError)
                    predictedVals = retArr.copy()
                    stump['inequal'] = inequal
                    stump['threshold'] = threshold
                    stump['dim'] = i
    return stump, minError, predictedVals


def adaBoostTrainDS(dataArr, labels, numIt):
    labelMat = mat(labels, dtype=float).T
    m = labelMat.shape[0]
    # D像个概率分布，因为其元素和为1
    D = ones((m, 1))/m
    # 各元素的预测值加权和
    classifyCount = zeros((m, 1))
    stumps = []
    for i in range(numIt):
        stump, error, retArr = buildStump(dataArr, labels, D)
        alpha = 0.5 * log((1.0 - error) / max(error, 1e-16))
        stump['alpha'] = alpha
        stumps.append(stump)
        # 重新计算D向量
        expon = multiply(-1 * alpha * retArr, labelMat)
        D_temp = multiply(D, exp(expon))
        D = D_temp / D_temp.sum()

        classifyCount += alpha * retArr
        z = zeros((m, 1))
        z[sign(classifyCount) != labelMat] = 1
        errorRate = z.sum()/m
        if errorRate == 0.0:
            # 如果所有元素已正确分类，则提前退出
            break

    return stumps


def adaClassify(stumps, testArr):
    testMat = mat(testArr, dtype=float)
    # 各元素的预测值加权和
    classifyCount = zeros((testMat.shape[0], 1))
    for i in range(len(stumps)):
        s = stumps[i]
        retArr = stumpClassify(testMat, s['dim'], s['threshold'], s['inequal'])
        classifyCount += s['alpha'] * retArr
    return sign(classifyCount)


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


def testing1():
    datMat, classLabels = loadSimpData()
    stumps = adaBoostTrainDS(datMat, classLabels, 10)
    print(adaClassify(stumps, [[0, 0]]))


def testing2():
    trainingDataMat, trainingClassLabels = loadDataSet('horseColicTraining2.txt')
    testDataMat, testClassLabels = loadDataSet('horseColicTest2.txt')
    stumps = adaBoostTrainDS(trainingDataMat, trainingClassLabels, 50)
    classify = adaClassify(stumps, testDataMat)
    m = len(testClassLabels)
    testLabelMat = mat(testClassLabels, dtype=float).T
    incorrect = zeros((m, 1))
    incorrect[testLabelMat != classify] = 1
    print('Error rate is ', float(incorrect.sum())/m)


#testing1()
testing2()

