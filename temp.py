import tensorflow as tf
import numpy as np


def test1():
    a = np.array([[0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 1., 0., 0., 1., 0., 1., 0.],
       [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
       [0., 0., 1., 0., 0., 0., 0., 0., 0., 1.],
       [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]])

    # 这段是固定的
    init = tf.global_variables_initializer()
    session = tf.Session()
    session.run(init)

    v = tf.argmax(a, axis=0)
    print(session.run(v))
    z2=np.mat([1, 2, 3], dtype=np.float32)
    print(session.run(tf.nn.softmax(z2)))

def test2():
    logits = tf.constant([[10.0, 4.0, 3.0],
                          [2.0, 26.0, 5.0],
                          [1.0, 2.0, 9.0]])
    y = tf.nn.softmax(logits, axis=0)
    # true label
    y_label = tf.constant([[1, 0, 0], [0, 1, 0], [0.0, 0.0, 1.0]])
    cross_entropy = -tf.reduce_sum(y_label * tf.log(y))
    cost = - tf.reduce_sum(tf.multiply(y_label, y))

    sess = tf.Session()
    print('y=', sess.run(y))
    mul = sess.run(tf.multiply(y_label, y))
    print('mul=', mul)

    mul2 = sess.run(y_label * y)
    print('mul2=', mul2)
    print('cost=', sess.run(cost))
    print('log=', sess.run(tf.log(y)))
    print('cross_entropy=', sess.run(cross_entropy))


test2()