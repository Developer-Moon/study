from sklearn.metrics import r2_score, accuracy_score, mean_absolute_error
import tensorflow as tf
tf.compat.v1.set_random_seed(123)


#1. 데이터
x_data = [[0, 0], [0, 1], [1, 0], [1, 1]] # (4, 2)
y_data = [[0], [1], [1], [0]]             # (4, 1)

# input later
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

# hidden layer
w1 = tf.compat.v1.Variable(tf.random_normal([2, 20]), name='weight') # 히든레이어를 20개 하려면
b1 = tf.compat.v1.Variable(tf.random_normal([20]), name='bias')      # bias도 20개

hidden_layer_01 = tf.compat.v1.matmul(x, w1) + b1

# hidden layer
w2 = tf.compat.v1.Variable(tf.random_normal([20, 1]), name='weight') # 두번째 층이 받아 들이는 것
b2 = tf.compat.v1.Variable(tf.random_normal([1]), name='bias')

hidden_layer_02 = tf.compat.v1.sigmoid(tf.matmul(hidden_layer_01, w2) + b2)

# output layer
w3 = tf.compat.v1.Variable(tf.random_normal([1, 1]), name='weight') # 두번째 층이 받아 들이는 것
b3 = tf.compat.v1.Variable(tf.random_normal([1]), name='bias')


#2. 모델구성
hypothesis = tf.compat.v1.sigmoid(tf.matmul(hidden_layer_02, w3) + b3)
                                                       


#3-1 컴파일                   
loss = -tf.reduce_mean(y * tf.log(hypothesis) + (1 - y) * tf.log(1 - hypothesis))
optimizer = tf.train.AdamOptimizer(learning_rate=1e-2) 
train = optimizer.minimize(loss)



with tf.compat.v1.Session() as sess : 
    sess.run(tf.global_variables_initializer())
    
    epochs = 1001
    for step in range(epochs):
        cost_val, hy_val, _ = sess.run([loss, hypothesis, train], feed_dict={x:x_data, y:y_data})
        if step %20 == 0:
            print(epochs, 'loss :', cost_val, '\n', hy_val)
   
    y_predict = sess.run(tf.cast(hy_val>=0.5, dtype=tf.float32)) # 이 값이 참이면 1 거짓이면 0
    # y_predict = x1_data * W1_val + x2_data *W2_val + x3_data * W3_val + b_val # r2 : 0.9538024379634447
    
    
    
    print('y예측 :', y_predict)

    acc = accuracy_score(y_data, y_predict)
    print('acc :', acc) 

