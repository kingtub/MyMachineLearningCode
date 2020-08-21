import tensorflow as tf

x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")
f = x * x * y + y + 2
#以上代码并没有实际计算， 甚至都没有初始化， 它只是创建一个图f

# 创建会话
sess = tf.Session()
# 初始化x变量
sess.run(x.initializer)
# 初始化y变量
sess.run(y.initializer)
# 执行图计算
result = sess.run(f)
print(result)

# 关闭会话
sess.close()
