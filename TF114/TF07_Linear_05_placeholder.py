# y = wx + b
import tensorflow as tf
tf.set_random_seed(123)


#1. 데이터
x = tf.placeholder(tf.float32, shape=[None]) # shape에 대해 모르면 None으로 잡으면 input_shape가 자동으로 잡힌다
y = tf.placeholder(tf.float32, shape=[None])

W = tf.Variable(tf.random_normal([1]), dtype=tf.float32) 
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32) 


#2. 모델구성
hypothesis = x * W + b


#3-1 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)


#3-2 훈련
with tf.compat.v1.Session() as sess : # sess = tf.compat.v1.Session() 대신 with문을 써서 마지막에 sess close()를 쓰지 않는다
    sess.run(tf.global_variables_initializer())
    
    epochs = 2001
    for step in range(epochs):
        # sess.run(train)
        _, loss_val, W_val, b_val = sess.run((train, loss, W, b), feed_dict={x:[1, 2, 3, 4, 5], y:[1, 2, 3, 4, 5]})
        # _, : 앞에 반환은 하지 않겠다 하지만 실행은 하겠다 _는 A라고 해도 되고 변수 같은거다 - 필요가 없어서 반환을 하지 않는다
        if step %20 == 0:
            # print(step, sess.run(loss), sess.run(W), sess.run(b))
            print(step, loss_val, W_val, b_val)