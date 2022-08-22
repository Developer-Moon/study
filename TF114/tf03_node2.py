import tensorflow as tf


node_01 = tf.constant(2.0)
node_02 = tf.constant(3.0)

# 덧셈 node_03
# 뺄셈 node_04 
# 곱셈 node_05
# 나눗셈 node_06


sess = tf.Session()

node_03 = tf.add(node_01, node_02)       # 더하기
node_03 = node_01 + node_02
print(sess.run(node_03)) # 5.0 


node_04 = tf.subtract(node_01, node_02)  # 빼기
node_04 = node_01 - node_02
print(sess.run(node_04)) # -1.0


node_05 = tf.multiply(node_01, node_02) # 원소곱
# node_05 = tf.matmul(node_01, node_02) # 행렬곱
print(sess.run(node_05)) # 6.0


node_06 = tf.divide(node_01, node_02) # tf.divide(~를, ~로 나누면?)      / 0.6666667
# node_06 = tf.mod(node_01, node_02) # tf.mod(~를, ~로 나눈 나머지는?)

print(sess.run(node_06)) 