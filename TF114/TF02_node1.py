import tensorflow as tf


node_01 = tf.constant(3.0, tf.float32) # flaot32 - 32bits 
node_02 = tf.constant(4.0)             # float64 - 64bits 메모리 용량차이 2배 연산속도 차이 32bits에 비해 정확하게 숫자를 나타낸다 정밀하게 작업할때 사용

node_03 = node_01 + node_02
node_03 = tf.add(node_01, node_02)   


sess = tf.compat.v1.Session()          # sess = tf.Session()
print(sess.run(node_03))               # 7.0