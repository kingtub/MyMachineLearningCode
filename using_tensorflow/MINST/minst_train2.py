import tensorflow as tf
import numpy as np

import minst_data




def my_train():
    # 这是一个只有输入层和输出层的神经网络
    # 1、定义变量
    layers = [784, 100, 10]
    w1 = tf.Variable(tf.random_uniform([100, 784], 0, 0.1))
    w2 = tf.Variable(tf.random_uniform([10, 100], 0, 0.1))
    # 偏置单元
    b1 = tf.Variable(tf.random_uniform([100, 1], 0, 0.1))
    b2 = tf.Variable(tf.random_uniform([10, 1], 0, 0.1))

    x = tf.placeholder(tf.float32, [784, None])
    y_label = tf.placeholder(tf.float32, [10, None])

    # 2、定义代价函数
    z1 = tf.matmul(w1, x) + b1
    a1 = tf.sigmoid(z1)
    z2 = tf.matmul(w2, a1) + b2
    y = tf.nn.log_softmax(z2, axis=0)
    cost = - tf.reduce_sum(y_label * y)

    # 3、定义梯度下降训练法，0.05是学习步长
    train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cost)

    # 这段是固定的
    init = tf.global_variables_initializer()
    session = tf.Session()
    session.run(init)

    data = minst_data.Data(100)

    # 训练30轮
    for i in range(2):
        for j in range(600):
            print((i,j))
            imgs, labels = data.next_train_batch()
            imgs = imgs.transpose()
            session.run(train_step, feed_dict={x: imgs, y_label: labels})
        data.re_init()


    # 测试
    test_images, test_labels = data.test_data()
    test_images = test_images.transpose()

    zt1 = tf.matmul(w1, test_images) + b1
    at1 = tf.sigmoid(zt1)
    zt2 = tf.matmul(w2, at1) + b2

    y2 = tf.nn.softmax(zt2, axis=0)

    # correct_prediction = tf.equal(tf.argmax(y, axis=0), tf.argmax(y_label, axis=0))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # print(session.run(accuracy, feed_dict = {x:test_images.transpose(), y_label:test_labels}))
    correct_prediction = tf.equal(tf.argmax(y2, axis=0), tf.argmax(test_labels, axis=0))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print(session.run(accuracy))


my_train()