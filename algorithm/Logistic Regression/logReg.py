from numpy import *
import matplotlib.pyplot as plt

# 激活函数
def sigmoid(inX):
    return 1 / (1.0 + exp(-inX))


def loadData():
    features = []
    labels = []
    lines = open('testSet.txt').readlines()
    for line in lines:
        li = line.strip().split('\t')
        features.append([1, li[0], li[1]])
        labels.append(li[2])
    return features, labels


# 通过梯度上升方法求线性系数
def gradientAscent(features, labels):
    labelsVec = mat(labels, dtype='float64')
    lm = labelsVec.shape[1]
    # 标签向量
    labelsVec = labelsVec.reshape(lm, 1)
    # 特征矩阵
    featuresMat = mat(features, dtype='float64')
    m, n = featuresMat.shape
    # 把线性系数全初始化为1
    weights = mat(ones((n, 1)))
    iter = 500 # 循环次数
    alpha = 0.001 # 学习步长
    for o in range(iter):
        # 偏导数d(L(w))/d(w_j)=(y - h_w(x))x_j
        # 梯度上升公式为：
        # w_j = w_j + alpha * sigma(i=1:m)(y_i - h_w_j(x_i))x_j_i
        # 参考 https://blog.csdn.net/c406495762/article/details/77723333
        # 这是向量化了的方式
        h = sigmoid(featuresMat * weights)
        error = labelsVec - h
        weights += alpha * featuresMat.transpose() * error
    return weights


# 随机梯度上升
def randomGradientAscent(features, labels):
    labelsVec = mat(labels, dtype='float64')
    lm = labelsVec.shape[1]
    # 标签向量
    labelsVec = labelsVec.reshape(lm, 1)
    # 特征矩阵
    featuresMat = mat(features, dtype='float64')
    m, n = featuresMat.shape
    # 把线性系数全初始化为1
    weights = mat(ones((n, 1)))
    iter = 150  # 循环次数
    alpha = 0.001  # 学习步长
    for o in range(iter):
        for i in range(m):
            x_i = featuresMat[i, :]
            h = sigmoid(x_i * weights)
            weights += alpha * float(labelsVec[i, 0] - h) * x_i.transpose()
    return weights

# 改进的随机梯度上升
# iter 循环次数
def randomGradientAscent2(features, labels, iter = 100):
    labelsVec = mat(labels, dtype='float64')
    lm = labelsVec.shape[1]
    # 标签向量
    labelsVec = labelsVec.reshape(lm, 1)
    # 特征矩阵
    featuresMat = mat(features, dtype='float64')
    m, n = featuresMat.shape
    # 把线性系数全初始化为1
    weights = mat(ones((n, 1)))
    for j in range(iter):
        indexs = list(range(m))
        for i in range(m):
            # 改进的地方1 - 动态修改 学习步长alpha
            alpha = 4 / (1 + j + i) + 0.01
            # 改进的地方2 - 随机选取样本
            d = int(random.uniform(0, len(indexs)))
            indexRandom = indexs[d]
            del indexs[d]
            x = featuresMat[indexRandom, :]
            h = sigmoid(x * weights)
            weights += alpha * float(labelsVec[indexRandom, 0] - h) * x.transpose()
    return weights


# 把结果 画出来
def drawResult(features, labels, weights, title):
    # 画图
    x = [float(e[1]) for e in features]
    y = [float(e[2]) for e in features]
    # plt.scatter(x, y) # 这才是画散点图

    # 通过画不同颜色的散点图即可知道哪些样本有没有被分类错，不用验证
    xred = [float(features[i][1]) for i in range(len(features)) if labels[i] is '1']
    yred = [float(features[i][2]) for i in range(len(features)) if labels[i] is '1']
    xblue = [float(features[i][1]) for i in range(len(features)) if labels[i] is '0']
    yblue = [float(features[i][2]) for i in range(len(features)) if labels[i] is '0']
    plt.scatter(xred, yred, c='r')
    plt.scatter(xblue, yblue, c='b')

    # 画分界函数线
    small = min(x)
    big = max(x)
    x1 = linspace(small, big, 100)
    x2 = -(weights[0, 0] + weights[1, 0] * x1) / weights[2, 0]
    plt.plot(x1, x2)

    plt.title(title)
    #plt.show()


def testing():
    features, labels = loadData()
    # print(features)
    # print(labels)

    weights0 = gradientAscent(features, labels)
    weights1 = randomGradientAscent(features, labels)
    weights2 = randomGradientAscent2(features, labels, 20)
    # 打印出学习得到的权重
    print('weights0', weights0)
    print('weights1', weights1)
    print('weights2', weights2)

    plt.figure(0)
    drawResult(features, labels, weights0, 'p0')
    plt.figure(1)
    drawResult(features, labels, weights1, 'p1')
    plt.figure(2)
    drawResult(features, labels, weights2, 'p2')
    plt.show()


# testing()

