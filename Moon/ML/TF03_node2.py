import tensorflow as tf


node_01 = tf.constant(2.0)
node_02 = tf.constant(3.0)

sess = tf.compat.v1.Session() # sess = tf.Session()

node_03 = tf.add(node_01, node_02)
node_03 = node_01 + node_02
print(sess.run(node_03))


node_04 = tf.subtract(node_01, node_02)
node_04 = node_01 - node_02
print(sess.run(node_04))


node_05 = tf.multiply(node_01, node_02) # 원소곱
node_05 = tf.matmul(node_01, node_02)   # 행렬곱
node_05 = node_01 * node_02             
print(sess.run(node_05))


node_06 = tf.divide(node_01, node_02)   # 나누기
node_06 = tf.mod(node_01, node_02)      # 나머지
node_06 = node_01 / node_02
node_06 = tf
print(sess.run(node_06))