import tensorflow as tf


print(tf.__version__)
print(tf.executing_eagerly())
tf.compat.v1.disable_eager_execution()
print(tf.executing_eagerly())

node_01 = tf.constant(3.0, tf.float32)
node_02 = tf.constant(4.0)
node_03 = tf.add(node_01, node_02)

sess = tf.compat.v1.Session()

a = tf.compat.v1.placeholder(tf.float32)
b = tf.compat.v1.placeholder(tf.float32)

add_node = a + b

print(sess.run(add_node, feed_dict={a:3, b:4.5}))
print(sess.run(add_node, feed_dicr={a:[1, 3], b:[2, 4]}))

add_and_triple = add_node * 3
print(sess.run(add_and_triple, feed_dict={a:3, b:4.5}))