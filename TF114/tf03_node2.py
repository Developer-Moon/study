import tensorflow as tf


node_01 = tf.constant(2.0)
node_02 = tf.constant(3.0)

sess = tf.Session() # sess = tf.compat.v1.Session()

node_03 = tf.add(node_01, node_02)      # 더하기
node_03 = node_01 + node_02
print(sess.run(node_03))                # 5.0 


node_04 = tf.subtract(node_01, node_02) # 빼기
node_04 = node_01 - node_02
print(sess.run(node_04))                # -1.0


node_05 = tf.multiply(node_01, node_02) # 원소곱
node_05 = node_01 * node_02
# node_05 = tf.matmul(node_01, node_02) # 행렬곱
print(sess.run(node_05))                # 6.0


node_06 = tf.divide(node_01, node_02)   # 나누기 / node_06 = tf.div(node_01, node_02)도 가능
node_06 = node_01 / node_02
# node_06 = tf.mod(node_01, node_02)    # 나머지 2
print(sess.run(node_06))                # 0.6666667