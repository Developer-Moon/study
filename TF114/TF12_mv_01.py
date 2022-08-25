from sklearn.metrics import r2_score, mean_absolute_error
import tensorflow as tf
tf.compat.v1.set_random_seed(123)


#1. 데이터
x1_data = [73., 93., 89., 96., 73]
x2_data = [80., 88., 91., 98., 66.]
x3_data = [75., 93., 90., 100., 70.]
y_data = [152. ,185., 180., 196., 142.]

x1 = tf.compat.v1.placeholder(tf.float32, shape=[None])
x2 = tf.compat.v1.placeholder(tf.float32, shape=[None])
x3 = tf.compat.v1.placeholder(tf.float32, shape=[None])
y = tf.compat.v1.placeholder(tf.float32, shape=[None])

w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), name='weight_01')
w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), name='weight_02')
w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), name='weight_03')
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), name='bias')



#2. 모델 
hypothesis = x1 * w1 + x2 *w2 + x3 * w3 + b



#3-1 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5) 
train = optimizer.minimize(loss)

#3-2

with tf.compat.v1.Session() as sess : 
    sess.run(tf.global_variables_initializer())
    
    epochs = 1001
    for step in range(epochs):
        # sess.run(train)
        _, loss_val, W1_val, W2_val, W3_val, b_val = sess.run((train, loss, w1, w2, w3, b), feed_dict={x1:x1_data, x2:x2_data, x3:x3_data, y:y_data})
        # _, : 앞에 반환은 하지 않겠다 하지만 실행은 하겠다 _는 A라고 해도 되고 변수 같은거다 - 필요가 없어서 반환을 하지 않는다
        if step %20 == 0:
            # print(step, sess.run(loss), sess.run(W), sess.run(b))
            print(step, loss_val, W1_val, W2_val, W3_val, b_val)

    
    y_predict = sess.run(hypothesis, feed_dict={x1:x1_data, x2:x2_data, x3:x3_data})
    # y_predict = x1_data * W1_val + x2_data *W2_val + x3_data * W3_val + b_val # r2 : 0.9538024379634447
    
    print('y예측 :', y_predict)

    r2 = r2_score(y_data, y_predict)
    print('r2 :', r2)

    mae = mean_absolute_error(y_data, y_predict)
    print('mae :', mae)



with tf.compat.v1.Session() as sess : 
    sess.run(tf.global_variables_initializer())
    
    epochs = 1001
    for step in range(epochs):
        # sess.run(train)
        cost_val, hy_val, _ = sess.run([loss, hypothesis, train], feed_dict={x1:x1_data, x2:x2_data, x3:x3_data, y:y_data})
        # _, : 앞에 반환은 하지 않겠다 하지만 실행은 하겠다 _는 A라고 해도 되고 변수 같은거다 - 필요가 없어서 반환을 하지 않는다
        if step %20 == 0:
            # print(step, sess.run(loss), sess.run(W), sess.run(b))
            print(epochs, 'loss :', cost_val, '\n', hy_val)
    
    
    y_predict = sess.run(hypothesis, feed_dict={x1:x1_data, x2:x2_data, x3:x3_data})
    # y_predict = x1_data * W1_val + x2_data *W2_val + x3_data * W3_val + b_val # r2 : 0.9538024379634447
    
    print('y예측 :', y_predict)

    r2 = r2_score(y_data, hy_val)
    print('r2 :', r2)

    mae = mean_absolute_error(y_data, hy_val)
    print('mae :', mae)
