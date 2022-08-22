import tensorflow as tf


node_01 = tf.constant(3.0, tf.float32)
node_02 = tf.constant(4.0)

node_03 = node_01 + node_02
node_03 = tf.add(node_01, node_02) # 위에꺼랑 같다


# print(node_03) 
# Tensor("add:0", shape=(), dtype=float32)
# sess = tf.Session()
sess = tf.compat.v1.Session()
print(sess.run(node_03))  # 7.0

sd