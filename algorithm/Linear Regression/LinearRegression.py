from numpy import *
import matplotlib.pyplot as plt


def loadData():
    xs = []
    ys = []
    lines = open('ex0.txt').readlines()
    for line in lines:
        li = line.strip().split('\t')
        xs.append([li[0], li[1]])
        ys.append(li[2])
    return array(xs, dtype=float), array(ys, dtype=float)


def train(xs, ys):
    weights = mat(ones((2, 1)))
    xMat = mat(xs)
    yMat = mat(ys).transpose()
    alpha = 0.001
    times = 10

    for i in range(times):
        error = xMat * weights - yMat
        ws = alpha * (((2 * error).transpose() * xMat).transpose())
        weights -= ws

    return weights


def trainTF(xs, ys):
    import tensorflow as tf

    xMat = mat(xs, dtype='float32')
    yMat = mat(ys, dtype='float32').transpose()
    alpha = 0.001
    times = 120

    W = tf.Variable(tf.random_uniform([2, 1], -1.0, 1.0))
    error = tf.matmul(xMat, W) - yMat
    cost = tf.reduce_sum(tf.square(error))
    # 3、定义梯度下降训练法，0.001是学习步长
    train = tf.train.GradientDescentOptimizer(alpha).minimize(cost)

    # 这段是固定的
    init = tf.global_variables_initializer()
    session = tf.Session()
    session.run(init)

    x_steps = []
    y_cost = []
    for i in range(times):
        session.run(train)
        if (i+1) % 10 == 0:
            x_steps.append(i+1)
            y_cost.append(session.run(cost))

    plt.figure(0)
    plt.plot(x_steps, y_cost)
    print('xc=',x_steps)
    print('yc=', y_cost)

    plt.figure(1)

    # 把最终变量结果打印出来
    weights_result = session.run(W)

    return weights_result


# 把结果 画出来
def drawResult(xs, ys, weights):
    # 画散点图
    plt.scatter(xs[:, 1], ys[:])


    # 画分界函数线
    small = min(xs[:, 1])
    big = max(xs[:, 1])
    x = linspace(small, big, 100)
    y = weights[0, 0] + weights[1, 0] * x
    plt.plot(x, y)

    plt.show()


def testing():
    xs, ys = loadData()
    # weights = train(xs, ys)
    weights = trainTF(xs, ys)
    print(weights)
    drawResult(xs, ys, weights)


testing()