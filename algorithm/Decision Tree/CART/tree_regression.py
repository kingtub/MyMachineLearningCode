from numpy import *

key_featIndex = 'featIndex'
key_featValue = 'featValue'
key_left_tree = 'left'
key_right_tree = 'right'


def loadDataSet(fileName):
    f = open(fileName)
    dataMat = []
    for line in f.readlines():
        lineArr = line.strip().split('\t')
        sample = []
        for s in lineArr:
            sample.append(float(s))
        dataMat.append(sample)
    return dataMat


# 回归树的叶节点取几个样本值的均值
def leafMean(dataSet):
    return mean(dataSet[:, -1])


# 误差取几个样本的总方差
def errVar(dataSet):
    return var(dataSet[:, -1]) * dataSet.shape[0]


# 把数据分成2份
def binSplitData(dataSet, featIndex, featValue):
    leftIndex = [i for i in range(dataSet.shape[0]) if dataSet[i, featIndex] > featValue]
    rightIndex = [i for i in range(dataSet.shape[0]) if dataSet[i, featIndex] <= featValue]
    leftTree = dataSet[leftIndex, :]
    rightTree = dataSet[rightIndex, :]
    return leftTree, rightTree


# 寻找此数据集dataSet上的最佳二元切分，使得2个子集上的方差之和是所有中最小的
def chooseBestSplit(dataSet, leafType, errType, tol):
    if dataSet[:, -1].tolist().count(dataSet[0, -1]) == dataSet.shape[0]:
        # 所有目标值一样
        return None, leafType(dataSet)
    tolS = tol[0]
    tolN = tol[1]
    S = errType(dataSet)
    m, n = dataSet.shape
    bestS = inf
    bestIndex = 0
    bestValue = 0
    for featIndex in range(n - 1):
        for featValue in set(dataSet[:, featIndex]):
            m0, m1 = binSplitData(dataSet, featIndex, featValue)
            if (m0.shape[0] < tolN) or (m1.shape[0] < tolN):
                # 子集样本数太少了，不用
                continue
            newS = errType(m0) + errType(m1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = featValue
                bestS = newS
    if bestS == inf:
        # 说明所分割子集都太小了
        return None, leafType(dataSet)
    m0Best, m1Best = binSplitData(dataSet, bestIndex, bestValue)
    if S - errType(m0Best) - errType(m1Best) < tolS:
        # 分割后带来的方差下降不够
        return None, leafType(dataSet)
    if (m0Best.shape[0] < tolN) or (m1Best.shape[0] < tolN):
        # 分割后子集样本数太少了，不用
        return None, leafType(dataSet)

    return bestIndex, bestValue


def createTree(dataSet, leafType=leafMean, errType=errVar, tol=(1, 4)):
    bestIndex, bestValue = chooseBestSplit(dataSet, leafType, errType, tol)
    if bestIndex == None:
        return bestValue
    leftSet, rightSet = binSplitData(dataSet, bestIndex, bestValue)
    tree = {}
    tree[key_featIndex] = bestIndex
    tree[key_featValue] = bestValue
    tree[key_left_tree] = createTree(leftSet, leafType, errType, tol)
    tree[key_right_tree] = createTree(rightSet, leafType, errType, tol)
    return tree


def predict(dTree, feature):
    if not isinstance(dTree, dict):
        # 单个值
        return dTree
    index = dTree[key_featIndex]
    value = dTree[key_featValue]
    if feature[index] > value:
        return predict(dTree[key_left_tree], feature)
    else:
        return predict(dTree[key_right_tree], feature)


# 树后剪枝过程 start
def isTree(a):
    return type(a).__name__ == 'dict'


def getMean(dTree):
    if isTree(dTree[key_left_tree]):
        dTree[key_left_tree] = getMean(dTree[key_left_tree])
    if isTree(dTree[key_right_tree]):
        dTree[key_right_tree] = getMean(dTree[key_right_tree])
    return (dTree[key_left_tree] + dTree[key_right_tree]) / 2


def prune(dTree, testData):
    '''
    后剪枝函数
    :param dTree: 用训练集数据得到的数
    :param testData: 用于剪枝的测试数据
    :return: 剪枝后的树
    '''
    if testData.shape[0] == 0:
        return getMean(dTree)
    if isTree(dTree[key_left_tree]) or isTree(dTree[key_right_tree]):
       lSet, rSet = binSplitData(testData, dTree[key_featIndex], dTree[key_featValue])
       if isTree(dTree[key_left_tree]):
           dTree[key_left_tree] = prune(dTree[key_left_tree], lSet)
       if isTree(dTree[key_right_tree]):
           dTree[key_right_tree] = prune(dTree[key_right_tree], rSet)
    if not isTree(dTree[key_left_tree]) and not isTree(dTree[key_right_tree]):
        lSet, rSet = binSplitData(testData, dTree[key_featIndex], dTree[key_featValue])
        # 如果树的两支都是叶子，现在测试合并与不合并的误差
        errorNoMerge = sum(power(lSet[:, -1] - dTree[key_left_tree], 2)) + sum(power(rSet[:, -1] - dTree[key_right_tree], 2))
        mean = (dTree[key_left_tree] + dTree[key_right_tree]) / 2
        errorMerge = sum(power(testData[:, -1] - mean, 2))
        if errorMerge < errorNoMerge:
            # 合并
            print("Merging")
            return mean
        else:
            return dTree
    return dTree

# 树后剪枝过程 end


def testing1():
    dataMat = loadDataSet('ex00.txt')
    tree = createTree(array(dataMat))
    print(tree)
    print('f(0.036098)=', predict(tree, [0.036098]))
    print('f(0.343554)=', predict(tree, [0.343554]))
    print('f(0.691115)=', predict(tree, [0.691115]))


def testing2():
    dataMat = loadDataSet('ex0.txt')
    tree = createTree(array(dataMat))
    print(tree)


def testingPrune():
    dataMat = loadDataSet('ex2.txt')
    dTree = createTree(array(dataMat), tol=(0, 1))
    print(dTree)
    testMat = loadDataSet('ex2test.txt')
    dTreePruned = prune(dTree, array(testMat))
    print(dTreePruned)


# testingPrune()
