import tensorflow as tf
from sklearn.metrics import r2_score, mean_absolute_error
tf.compat.v1.set_random_seed(66)

x_data = [[73, 51, 65],                         # (5, 3)
          [92, 98, 11],
          [89, 31, 33],
          [99, 33, 100],
          [17, 66, 79]]
y_data = [[152], [185], [180], [205], [142]]    # (5, 1) 

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 3]) # 행렬 연산에서는 shape명시를 한다
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([3, 1]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), name='bias')
# 곱해질 w의 행값
# (5, 3) * (3, 1) = (5, 1)


#2 모델구성
hypothesis = tf.compat.v1.matmul(x, w) + b # 이 연산을 하기 위해 w의 shape을 맞춰준다



#3-1 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-6) 
train = optimizer.minimize(loss)



with tf.compat.v1.Session() as sess : 
    sess.run(tf.global_variables_initializer())
    
    epochs = 1001
    for step in range(epochs):
        # sess.run(train)
        cost_val, hy_val, _ = sess.run([loss, hypothesis, train], feed_dict={x:x_data, y:y_data})
        # _, : 앞에 반환은 하지 않겠다 하지만 실행은 하겠다 _는 A라고 해도 되고 변수 같은거다 - 필요가 없어서 반환을 하지 않는다
        if step %20 == 0:
            # print(step, sess.run(loss), sess.run(W), sess.run(b))
            print(epochs, 'loss :', cost_val, '\n', hy_val)
    
   
    y_predict = sess.run(hypothesis, feed_dict={x:x_data, y:y_data})
    # y_predict = x1_data * W1_val + x2_data *W2_val + x3_data * W3_val + b_val # r2 : 0.9538024379634447
    
    print('y예측 :', y_predict)

    r2 = r2_score(y_data, hy_val)
    print('r2 :', r2)

    mae = mean_absolute_error(y_data, hy_val)
    print('mae :', mae)
    
