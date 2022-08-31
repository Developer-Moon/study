from sklearn.metrics import r2_score, accuracy_score, mean_absolute_error
import tensorflow as tf
tf.compat.v1.set_random_seed(123)


#1. 데이터
x_data = [[0, 0], [0, 1], [1, 0], [1, 1]] # (4, 2)
y_data = [[0], [1], [1], [0]]             # (4, 1)


#2. 모델구성
# input layer
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

# hidden layer
w1 = tf.compat.v1.Variable(tf.random_normal([2, 20]), name='weight') # 히든레이어를 20개 하려면
b1 = tf.compat.v1.Variable(tf.random_normal([20]), name='bias')      # bias도 20개

hidden_layer_01 = tf.compat.v1.matmul(x, w1) + b1

# output layer
w2 = tf.compat.v1.Variable(tf.random_normal([20, 1]), name='weight') # 두번째 층이 받아 들이는 것
b2 = tf.compat.v1.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.compat.v1.sigmoid(tf.matmul(hidden_layer_01, w2) + b2)


#3-1 컴파일                   
loss = -tf.reduce_mean(y * tf.log(hypothesis) + (1 - y) * tf.log(1 - hypothesis))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01) 
train = optimizer.minimize(loss)


with tf.compat.v1.Session() as sess : 
    sess.run(tf.global_variables_initializer())
    
    epochs = 501
    for step in range(epochs):
        cost_val, hy_val, _ = sess.run([loss, hypothesis, train], feed_dict={x:x_data, y:y_data})
        if step % 20 == 0 :
            print(epochs, 'loss :', cost_val, '\n', hy_val)
   
    y_predict = sess.run(tf.cast(hy_val>=0.5, dtype=tf.float32)) # 이 값이 참이면 1 거짓이면 0

    acc = accuracy_score(y_data, y_predict)
    print('acc :', acc) 

    # acc : 0.75