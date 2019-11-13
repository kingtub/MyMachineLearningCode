import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

# 这是逻辑回归-用TensorFlow框架来实现
# 激活函数
def sigmoid(inX):
    return 1 / (1.0 + tf.exp(-inX))


def loadData():
    features = []
    labels = []
    lines = open('testSet.txt').readlines()
    for line in lines:
        li = line.strip().split('\t')
        features.append([1, li[0], li[1]])
        labels.append(li[2])
    return features, labels


def myTrain(features, labels):
    # 特征矩阵
    featuresMat = np.mat(features, dtype='float32')
    # 标签向量
    labelsVec = np.mat(labels, dtype='float32').transpose()

    # 1、定义变量
    w = tf.Variable(tf.random_uniform([3, 1], -1.0, 1.0))
    h_wx = sigmoid(tf.matmul(featuresMat, w))
    # 2、定义代价函数
    cost = -(tf.matmul(labelsVec.transpose(), tf.log(h_wx)) + tf.matmul((1 - labelsVec).transpose(), tf.log(1 - h_wx)))
    # 3、定义梯度下降训练法，0.001是学习步长
    train = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

    # 这段是固定的
    init = tf.global_variables_initializer()
    session = tf.Session()
    session.run(init)
    # 先打印初值
    #print(session.run(w))

    # 训练500次
    for i in range(500):
        session.run(train)

    # 把最终变量结果打印出来
    weights = session.run(w)
    print(weights)
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
    x1 = np.linspace(small, big, 100)
    x2 = -(weights[0, 0] + weights[1, 0] * x1) / weights[2, 0]
    plt.plot(x1, x2)

    plt.title(title)
    plt.show()


def testing():
    features, labels = loadData()
    weights = myTrain(features, labels)
    drawResult(features, labels, weights, 'p0')


    # weights = np.mat([[ 3.8953044], 这是得到的权值
    #                   [ 0.460072 ],
    #                   [-0.5880611]])
    # drawResult(features, labels, weights, 'p0')



testing()