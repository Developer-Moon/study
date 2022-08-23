import tensorflow as tf
tf.set_random_seed(123)

x = [1, 2, 3]
y = [1, 2, 3]

W = tf.Variable(1, dtype=tf.float32)
b = tf.Variable(1, dtype=tf.float32)

hypothesis = x * W + b

loss = tf.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

epochs = 2001

for step in range(epochs):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(loss), sess.run(W), sess.run(b))
        
sess.close()        