import tensorflow as tf
import numpy as np

# 神经网络用于二分类问题
# 这是一个2层神经网络，即输入特征为2个单元，隐藏层为4个单元，输出层为1个单元，以上不包括偏置单元
# 激活函数
def sigmoid(inX):
    return 1 / (1.0 + tf.exp(-inX))


def loadData():
    features = []
    labels = []
    lines = open('testSet.txt').readlines()
    for line in lines:
        li = line.strip().split('\t')
        features.append([li[0], li[1]])
        labels.append(li[2])
    return features, labels

def myTrain(features, labels, activate_func=sigmoid):
    # 特征矩阵
    featuresMat = np.mat(features, dtype='float32')
    # 标签向量
    labelsVec = np.mat(labels, dtype='float32').transpose()

    # 这是一个2层神经网络，即输入特征为2个单元，隐藏层为4个单元，输出层为1个单元，以上不包括偏置单元
    # 1、定义变量
    w1 = tf.Variable(tf.random_uniform([4, 2], 0, 0.1))
    # 偏置单元
    b1 = tf.Variable(tf.random_uniform([4, 1], 0, 0.1))
    w2 = tf.Variable(tf.random_uniform([1, 4], 0, 0.1))
    b2 = tf.Variable(tf.random_uniform([1, 1], 0, 0.1))
    # 2、定义代价函数
    z1 = tf.matmul(w1, featuresMat.transpose()) + b1
    a1 = activate_func(z1)
    z2 = tf.matmul(w2, a1) + b2
    # result
    a2 = activate_func(z2)

    cost = tf.reduce_sum(tf.square(a2 - labelsVec.transpose()), axis=1)
    # 3、定义梯度下降训练法，0.001是学习步长
    train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    # 这段是固定的
    init = tf.global_variables_initializer()
    session = tf.Session()
    session.run(init)

    # 训练10000次
    for i in range(10000):
        session.run(train)

    # 测试数据，若结果小于0.5，则为0，否则为1
    testVec = np.mat([[0.3, 11], [1.1, 8], [-0.2, 1.6], [1.7, 6.3]], dtype='float32').transpose()
    #z1 = tf.matmul(w1, featuresMat.transpose()) + b1
    z1 = tf.matmul(w1, testVec) + b1
    a1 = activate_func(z1)
    z2 = tf.matmul(w2, a1) + b2
    # result
    a2 = activate_func(z2)
    print(session.run(a2))


def testing():
    features, labels = loadData()
    myTrain(features, labels)


testing()