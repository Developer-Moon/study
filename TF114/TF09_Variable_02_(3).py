import tensorflow as tf
tf.set_random_seed(123)

# variable.eval()

#1. 데이터
x_train_data = [1, 2, 3]
y_train_data = [3, 5, 7]

x_train = tf.placeholder(tf.float32, shape=[None]) # shape에 대해 모르면 None으로 잡으면 input_shape가 자동으로 잡힌다
y_train = tf.placeholder(tf.float32, shape=[None])

W = tf.Variable(tf.random_normal([1]), dtype=tf.float32) 
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32) 

sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer())
W1 = W.eval()
b1 = b.eval()
sess.close()



#2. 모델구성
hypothesis = x_train * W + b



#3-1 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y_train))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.171)
train = optimizer.minimize(loss)



#3-2 훈련
sess = tf.compat.v1.InteractiveSession()
sess.run(tf.global_variables_initializer())

epochs = 101
for step in range(epochs):
    # sess.run(train)
    _, loss_val, W_val, b_val = sess.run((train, loss, W, b), feed_dict={x_train:x_train_data, y_train:y_train_data})
    # _, : 앞에 반환은 하지 않겠다 하지만 실행은 하겠다 _는 A라고 해도 되고 변수 같은거다 - 필요가 없어서 반환을 하지 않는다
    if step % 20 == 0:
        # print(step, sess.run(loss), sess.run(W), sess.run(b))
        print(step, loss_val, W_val, b_val)

x_test_data = [6, 7, 8]       
x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])

y_predict = x_test * W_val + b_val # y_predict = model.predict(x_test)

print('[6, 7, 8] 예측 :', sess.run(y_predict, feed_dict={x_test:x_test_data}))