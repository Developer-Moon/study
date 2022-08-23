import tensorflow as tf


print(tf.__version__)
print(tf.executing_eagerly())

tf.compat.v1.disable_eager_execution()
print(tf.executing_eagerly())

hello = tf.compat.v1.constant('hello')

sess = tf.compat.v1.Session()
print(sess.run(hello))