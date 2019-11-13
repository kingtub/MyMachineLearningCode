from numpy import *
from tree_regression import *

'''
模型树就是叶子节点是线性模型的树
'''

def linearSolve(dataSet):
    m, n = dataSet.shape
    x = ones((m, n))
    y = zeros((m, 1))
    # 最后一列是y值，另外每个样本都加1作为偏置变量
    x[:, 1:n] = dataSet[:, 0:n - 1]
    y = dataSet[:, n-1]
    mX = mat(x)
    mY = mat(y).transpose()
    xTx = mX.transpose() * mX
    if linalg.det(xTx) == 0:
        print('不可逆方阵，极小概率事件')
        return
    ws = linalg.inv(xTx) * mX.transpose() * mY
    return ws, mX, mY


def modelLeaf(dataSet):
    ws, x, y = linearSolve(dataSet)
    return ws


def modelErr(dataSet):
    ws, x, y = linearSolve(dataSet)
    yHat = x * ws
    return sum(power((y - yHat), 2))


def modelPredict(dTree, feature):
    if not isinstance(dTree, dict):
        # 单个叶子
        # todo
        # ff = mat([[1].extend(feature)])
        # print(ff)
        return dot([feature], dTree)
    index = dTree[key_featIndex]
    value = dTree[key_featValue]
    if feature[index] > value:
        return modelPredict(dTree[key_left_tree], feature)
    else:
        return modelPredict(dTree[key_right_tree], feature)


def testingModelTree():
    dataMat = loadDataSet('exp2.txt')
    tree = createTree(array(dataMat), leafType=modelLeaf, errType=modelErr, tol=(1, 10))
    # todo
    # print('f(0.747221)=', modelPredict(tree, [0.747221]))
    # print('f(0.586082)=', modelPredict(tree, [0.586082]))
    # print('f(0.346734)=', modelPredict(tree, [0.346734]))
    print('f(0.747221)=', modelPredict(tree, [1, 0.747221]))
    print('f(0.586082)=', modelPredict(tree, [1, 0.586082]))
    print('f(0.346734)=', modelPredict(tree, [1, 0.346734]))
    #print(tree)


testingModelTree()
