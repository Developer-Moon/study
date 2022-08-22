import tensorflow as tf
print(tf.__version__) # 1.14.0

# print('hello world')


hello = tf.constant('hello world') # constant : 상수 고정값
print(hello) # Tensor("Const:0", shape=(), dtype=string)

# sess = tf.Session()
sess = tf.compat.v1.Session()
print(sess.run(hello))s