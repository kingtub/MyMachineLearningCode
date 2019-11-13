import tensorflow as tf


def test1():
    node1 = tf.constant(3.0, tf.float32)
    node2 = tf.constant(4.0)  # also tf.float32 implicitly
    print(node1, node2)
    # 最终的打印声明生成
    # Tensor("Const:0", shape=(), dtype=float32)
    # Tensor("Const_1:0", shape=(), dtype=float32)
    # 请注意，打印节点不会输出值3.0，4.0正如您所期望的那样。相反，它们是在评估时分别产生3.0和4.0的节点。要实际评估节点，我们必须在会话中运行计算图。

    # 下面的代码创建一个Session对象，然后调用其run方法运行足够的计算图来评价node1和node2。通过在会话中运行计算图如下：
    sess = tf.Session()
    print(sess.run([node1, node2]))

    # 我们可以通过将Tensor节点与操作相结合来构建更复杂的计算（操作也是节点）。例如，我们可以添加我们的两个常量节点并生成一个新的图，如下所示：
    node3 = tf.add(node1, node2)
    print("node3: ", node3)
    print("sess.run(node3): ", sess.run(node3))


def test2():
    sess = tf.Session()
    a = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)
    adder_node = a + b  # + provides a shortcut for tf.add(a, b)

    print(sess.run(adder_node, {a: 3, b: 4.5}))
    print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))

    add_and_triple = adder_node * 3.
    print(sess.run(add_and_triple, {a: 3, b: 4.5}))


def test3():
    sess = tf.Session()
    W = tf.Variable([.3], tf.float32)
    b = tf.Variable([-.3], tf.float32)
    x = tf.placeholder(tf.float32)
    linear_model = W * x + b

    # 常数被调用时初始化tf.constant，其值永远不会改变。相比之下，调用时，变量不会被初始化tf.Variable。
    # 要初始化TensorFlow程序中的所有变量，必须显式调用特殊操作，如下所示：
    init = tf.global_variables_initializer()
    sess.run(init)

    print(sess.run(linear_model, {x: [1, 2, 3, 4]}))

    y = tf.placeholder(tf.float32)
    squared_deltas = tf.square(linear_model - y)
    loss = tf.reduce_sum(squared_deltas)
    print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

    fixW = tf.assign(W, [-1.])
    fixb = tf.assign(b, [1.])
    sess.run([fixW, fixb])
    print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

    # train
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)

    sess.run(init)  # reset values to incorrect defaults.
    for i in range(1000):
        sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})

    print(sess.run([W, b]))


def train_():
    # Model parameters
    W = tf.Variable([.3], tf.float32)
    b = tf.Variable([-.3], tf.float32)
    # Model input and output
    x = tf.placeholder(tf.float32)
    linear_model = W * x + b
    y = tf.placeholder(tf.float32)
    # loss
    loss = tf.reduce_sum(tf.square(linear_model - y))  # sum of the squares
    # optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)
    # training data
    x_train = [1, 2, 3, 4]
    y_train = [0, -1, -2, -3]
    # training loop
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)  # reset values to wrong
    for i in range(1000):
        sess.run(train, {x: x_train, y: y_train})

    # evaluate training accuracy
    curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
    print("W: %s b: %s loss: %s" % (curr_W, curr_b, curr_loss))


# test3()
train_()
