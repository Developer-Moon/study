import tensorflow as tf


print(tf.__version__)
print(tf.executing_eagerly())           # True
tf.compat.v1.disable_eager_execution()  # 실행모드 꺼라
print(tf.executing_eagerly())           # False

node_01 = tf.constant(3.0, tf.float32)
node_02 = tf.constant(4.0)
node_03 = tf.add(node_01, node_02)

sess = tf.compat.v1.Session()

a = tf.compat.v1.placeholder(tf.float32) # placeholder 사용하여 a라는 공간을 만들어 input에 넣는다
b = tf.compat.v1.placeholder(tf.float32) # placeholder 사용하여 b라는 공간을 만들어 input에 넣는다


add_node = a + b

print(sess.run(add_node, feed_dict={a:3, b:4.5}))         # feed_dict로 딕셔너리 형태로 넣어준다 7.5
print(sess.run(add_node, feed_dict={a:[1, 3], b:[2, 4]})) # 행렬도 만들 수 있다  [3. 7.] 


add_and_triple = add_node * 3
print(add_and_triple) # Tensor("mul:0", dtype=float32)

print(sess.run(add_and_triple, feed_dict={a:3, b:4.5}))   # 22.5