from numpy import *
# K-近邻算法

debug = False


def using():
    # 从文本文件解析样本数据，得特征矩阵和相应的标签列表
    dataMat, lablesVector = matrixFromFile('datingTestSet2.txt')
    # 数值归一化，使得所有数据介于0、1之间
    normArr, minValues, ranges = normalize(dataMat)

    while input('using kNN? Y or N:') is not 'N':
        v1 = int(input("Enter kms:"))
        v2 = float(input('Enter ice cream:'))
        v3 = float(input('Enter ratio game time:'))
        # 先把得到的特征数值归一化
        di = (array([v1, v2, v3]) - minValues) / ranges
        gotLabel = classify(di, normArr, lablesVector, 3)
        print('label is', gotLabel)


def test():
    dataMat, lablesVector = matrixFromFile('datingTestSet2.txt')
    # print(dataMat)
    # print(lablesVector)
    # 用10%的数据作为测试数据，其余90%作为样本数据
    ratio = 0.1
    m = len(dataMat)
    start = int(m * ratio)
    if debug:
        print('start=', start)

    normArr, minValues, ranges = normalize(dataMat[start:m, :])

    if debug:
        print('normArr is ', normArr)
        print('minValues is ', minValues)
        print('ranges is ', ranges)
    errorCount = 0
    for i in range(start):
        di = (dataMat[i, :] - minValues) / ranges
        if debug:
            print('original v is ', dataMat[i, :])
            print('new v is ', di)

        gotLabel = classify(di, normArr, lablesVector[start:m], 3)
        if gotLabel != lablesVector[i]:
            errorCount += 1
            print('labelsVector[{}] is '.format(i), lablesVector[i], 'and predict is', gotLabel)
    print('Error rate is ', errorCount/start)

# 从文件加载数据
def matrixFromFile(fileName):
    f = open(fileName)
    lines = f.readlines()
    m = len(lines)
    dataMat = zeros((m, 3))
    lablesVector = []

    index = 0
    for lin in lines:
        nline = lin.strip()
        featuresListOneLine = nline.split('\t')
        dataMat[index, :] = featuresListOneLine[0: 3]
        lablesVector.append(int(featuresListOneLine[-1]))
        index += 1

    return dataMat, lablesVector


# 数值归一化
# newValue = （oldValue - minValue）/(maxValue - minValue)
def normalize(dataarr):
    minValues = dataarr.min(axis=0)
    maxValues = dataarr.max(axis=0)
    if debug:
        print('maxValues is ', maxValues)
    m = dataarr.shape[0]
    ranges = maxValues - minValues
    normArr = dataarr - tile(minValues, (m, 1))
    normArr = normArr / tile(ranges, (m, 1))
    return normArr, minValues, ranges


def classify(inV, dataMat, labelsV, k):
    m = dataMat.shape[0]
    arrb = dataMat - tile(inV, (m, 1))
    arrb **= 2
    distances = (arrb.sum(axis=1)) ** 0.5
    sortedDistanceIndices = distances.argsort()
    count = {}
    for i in range(k):
        label = labelsV[sortedDistanceIndices[i]]
        n = count.get(label, 0)
        count[label] = n + 1

    k0 = 1
    v0 = -5
    for k, v in count.items():
        if v > v0:
            k0 = k
            v0 = v
    return k0


# test()
using()
