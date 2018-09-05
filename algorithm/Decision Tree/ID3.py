from math import log
# 决策树 - ID3算法

# 本文件基于2维列表的数据，且最后一列是标签
def createTree(dataSet, labels):
    dataLabels = [example[-1] for example in dataSet]
    if dataLabels.count(dataLabels[0]) == len(dataLabels):
        # 所有样本标签都一样了，无需再分
        return dataLabels[0]
    elif len(dataSet[0]) == 1:
        # 只有一列特征了（即标签列）,投票，选择多数的一种
        return findMaxCountLabel(dataSet)

    # 根据最佳分割特征
    axis = chooseBestFeatureAxis(dataSet)
    topLabel = labels[axis]
    tree = {topLabel: {}}
    # 取出这特征列
    axisValue = [example[axis] for example in dataSet]
    # 唯一化
    aSet = set(axisValue)
    copyLabels = labels[:]
    del copyLabels[axis]

    for value in aSet:
        aData = splitDataSet(dataSet, axis, value)
        tree[topLabel][value] = createTree(aData, copyLabels)

    return tree


# 计算多数的一种（投票）
def findMaxCountLabel(labels):
    count = {}
    for label in labels:
        n = count.get(label, 0)
        count[label] = n + 1

    k0 = 1
    v0 = -5
    for k, v in count.items():
        if v > v0:
            k0 = k
            v0 = v
    return k0

# 计算给定数据集的 香农熵
def calShannonEnt(dataSet):
    num = len(dataSet)
    countDict = {}
    for vec in dataSet:
        la = vec[-1]
        oldValue = countDict.get(la, 0)
        countDict[la] = oldValue + 1

    ent = 0
    for k, v in countDict.items():
        prob = v / float(num)
        ent -= prob * log(prob, 2)

    return ent


# 按照给定数据集、维度和特征值划分数据
def splitDataSet(dataSet, axis, value):
    nlist = []
    for vec in dataSet:
        if vec[axis] == value:
            v1 = vec[0:axis]
            v1.extend(vec[axis + 1:])
            nlist.append(v1)

    return nlist

# 选择最好的分类维度
# 按照获取最大信息增益的方法划分数据集
def chooseBestFeatureAxis(dataSet):
    numFeatures = len(dataSet[0]) - 1
    numEntries = len(dataSet)
    entOriginal = calShannonEnt(dataSet)
    smallestEnt = entOriginal
    smallestFeature = -5
    for i in range(numFeatures):
        entI = 0
        # 提取全部此列特征
        aFeatures = [example[i] for example in dataSet]
        # 用集合来获取唯一值
        aSet = set(aFeatures)
        # 用每个值来分离数据，然后计算各自的香农熵，并按频率求总香农熵期望值
        for f in aSet:
            aData = splitDataSet(dataSet, i, f)
            prob = float(len(aData))/numEntries
            entI += prob * calShannonEnt(aData)
        if entOriginal - entI > entOriginal - smallestEnt:
            smallestEnt = entI
            smallestFeature = i

    return smallestFeature


# 分类函数
def classify(featureVec, labels, tree):
    la = list(tree.keys())[0]
    i = labels.index(la)
    subTree = tree[la]
    if not isinstance(subTree, dict):
        return subTree
    while isinstance(subTree[featureVec[i]], dict):
        la = list(subTree[featureVec[i]].keys())[0]
        subTree = subTree[featureVec[i]][la]
        i = labels.index(la)
    return subTree[featureVec[i]]


def saveTree(fileName, tree):
    import pickle
    with open(fileName, 'w') as fw:
        pickle.dump(fw, tree)


def loadTree(fileName):
    import pickle
    with open(fileName) as fr:
        return pickle.load(fr)


def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels


def test1():
    dataSet, labels = createDataSet()
    # print(dataSet, labels)
    tree = createTree(dataSet, labels)
    print(tree)

    print(classify([1, 1], labels, tree))
    print(classify([1, 0], labels, tree))
    print(classify([0, 1], labels, tree))


def test2():
    dataMat = []
    with open('lenses.txt') as f:
        for line in f.readlines():
            ss = line.strip().split('\t')
            dataMat.append(ss)
    labels = ['k1', 'k2', 'k3', 'k4']
    tree = createTree(dataMat, labels)
    print(classify(['young', 'myope', 'no', 'normal'], labels, tree))
    print(classify(['young', 'myope', 'no', 'reduced'], labels, tree))
    print(classify(['pre', 'myope', 'yes', 'normal'], labels, tree))


#test1()
test2()








