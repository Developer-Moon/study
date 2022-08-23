import tensorflow as tf


node_01 = tf.constant(3.0, tf.float32)
node_02 = tf.constant(4.0)

node_03 = node_01 + node_02

sess = tf.compat.v1.Session() # sess = tf.Session()
print(sess.run(node_03))