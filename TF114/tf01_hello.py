import tensorflow as tf


print(tf.__version__) # 1.14.0
print('hello world')


hello = tf.constant('hello world') # constant : 상수 고정값 - constant라는 노드 정의
print(hello)                       # Tensor("Const:0", shape=(), dtype=string)


sess = tf.compat.v1.Session()      # sess = tf.Session()
print(sess.run(hello))             # 텐서1은 출력을 할 때 반드시 sess.run() 을 거쳐야한다