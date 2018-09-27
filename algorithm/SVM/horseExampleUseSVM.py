from sklearn import svm


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

    clf = svm.SVC()  # class
    clf.fit(features, labels)
    # print('support_vectors_=', clf.support_vectors_)  # support vectors
    # print('support_=',clf.support_) # indeices of support vectors
    # print('n_support_=', clf.n_support_)  # number of support vectors for each class

    testFeatures = []
    testLabels = []
    for line in testFile.readlines():
        arr = []
        a = line.strip().split('\t')
        for i in range(21):
            arr.append(float(a[i]))
        testFeatures.append(arr)
        testLabels.append(float(a[21]))

    errorCount = 0
    for i in range(len(testFeatures)):
        if clf.predict([testFeatures[i]]) != [testLabels[i]]:
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