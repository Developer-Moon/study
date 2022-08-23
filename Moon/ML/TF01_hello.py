import tensorflow as tf


print(tf.__version__)
print('hello world')


hello = tf.constant('hello world')
print(hello)


sess = tf.compat.v1.Session() # sess = tf.Session()
print(sess.run(hello))