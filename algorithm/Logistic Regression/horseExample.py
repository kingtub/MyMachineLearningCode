from logReg import *


def exam(prob):
    if prob > 0.5:
        return 1
    else:
        return 0

def test():
    traningFile = open('horseColicTraining.txt')
    testFile = open('horseColicTest.txt')

    features = []
    labels = []
    for line in traningFile.readlines():
        arr = []
        a = line.strip().split('\t')
        for i in range(21):
            arr.append(float(a[i]))
        features.append(arr)
        labels.append(float(a[21]))

    weights = randomGradientAscent2(features, labels, 500)

    testFeatures = []
    testLabels = []
    for line in testFile.readlines():
        arr = []
        a = line.strip().split('\t')
        for i in range(21):
            arr.append(float(a[i]))
        testFeatures.append(arr)
        testLabels.append(float(a[21]))

    testFeaturesMat = mat(testFeatures)
    errorCount = 0
    for i in range(len(testFeatures)):
        r = testFeaturesMat[i, :] * weights
        # print(type(r))
        # print(r)
        prob = sigmoid(r)
        if exam(prob) != testLabels[i]:
            errorCount += 1

    return errorCount/len(testFeatures)


def multiTest():
    times = 10
    errorRateSum = 0
    for i in range(times):
        errorRate = test()
        errorRateSum += errorRate
        print('ErrorRate=', errorRate)
    print('average ErrorRate=', errorRateSum / times)


multiTest()