import tensorflow as tf
import matplotlib.pyplot as plt
tf.set_random_seed(123)


x_train = [1, 2, 3]
y_train = [1, 2, 3]
x = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), name='weight')
# w = tf.compat.v1.Variable([10], dtype=tf.float32, name='weight')


hypothesis = x * w

loss = tf.reduce_mean(tf.square(hypothesis - y)) 

lr = 0.1
gradient = tf.reduce_mean((w * x - y) * x) # GradientDescentOptimizer
descent = w - lr * gradient
update = w.assign(descent)

w_history = []
loss_history =[]

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(21) :
    _, loss_v, w_v = sess.run([update, loss, w], feed_dict={x:x_train, y:y_train}) 
    print(step, '\t', loss_v, '\t', w_v) 
    
    w_history.append(w_v[0])
    loss_history.append(loss_v)
    
sess.close()

print('__________________________ W history __________________________')
print(w_history)
print('__________________________ loss history __________________________')
print(loss_history)