import tensorflow as tf
import numpy as np

import minst_data
import matplotlib.pyplot as plt

def my_train():
    # 这是一个只有输入层和输出层的神经网络
    # 1、定义变量
    w = tf.Variable(tf.random_uniform([10, 784], 0, 0.1))
    # 偏置单元
    b = tf.Variable(tf.random_uniform([10, 1], 0, 0.1))

    x = tf.placeholder(tf.float32, [784, None])
    y_label = tf.placeholder(tf.float32, [10, None])

    # 2、定义代价函数
    z1 = tf.matmul(w, x) + b # (10 * 784) * (784 * m) + (10 * 1)=(10 * m)
    y = tf.nn.log_softmax(z1, axis=0) # 用softmax，再log时容易因log(0)而得NaN的结果
    cost = tf.reduce_mean(-tf.reduce_sum(y_label * y))
    # 以下这行可以代表上面2行
    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_label, logits=z1, dim=0))

    # 3、定义梯度下降训练法，0.05是学习步长
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    # 这段是固定的
    init = tf.global_variables_initializer()
    session = tf.Session()
    session.run(init)

    data = minst_data.Data(100)

    print('w=', session.run(w))
    print('b=', session.run(b))
    x_steps = []
    y_cost = []
    # 训练1000次
    for i in range(1000):
        imgs, labels = data.next_train_batch()
        imgs = imgs.transpose()
        session.run(train_step, feed_dict={x: imgs, y_label: labels})
        # data.re_init()
        if(i + 1) % 50 == 0:
            x_steps.append(i+1)
            y_cost.append(session.run(cost, feed_dict={x: imgs, y_label: labels}))

    print('w=', session.run(w))
    print('b=', session.run(b))
    # 测试
    test_images, test_labels = data.test_data()
    test_images = test_images.transpose()
    z2 = tf.matmul(w, test_images) + b
    y2 = tf.nn.softmax(z2, axis=0)

    # correct_prediction = tf.equal(tf.argmax(y, axis=0), tf.argmax(y_label, axis=0))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # print(session.run(accuracy, feed_dict = {x:test_images.transpose(), y_label:test_labels}))
    correct_prediction = tf.equal(tf.argmax(y2, axis=0), tf.argmax(test_labels, axis=0))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print(session.run(accuracy))

    print('xc=', x_steps)
    print('yc=', y_cost)
    plt.plot(x_steps, y_cost)
    plt.show()


my_train()