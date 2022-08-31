from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
import tensorflow as tf
import numpy as np
tf.compat.v1.set_random_seed(123)


#1. 데이터
datasets = load_boston()
x_data = datasets.data
y_data = datasets.target


print(x_data.shape) # (506, 13)
print(y_data.shape) # (506, )
print(np.unique(y_data, return_counts=True)) # (array([0, 1]), array([212, 357], dtype=int64))
y_data = y_data.reshape(-1, 1)
print(y_data.shape) # (506, 1)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.75, random_state=123)

# input later
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 13])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])


# hidden layer
w1 = tf.compat.v1.Variable(tf.zeros([13, 20]), name='weight')
b1 = tf.compat.v1.Variable(tf.zeros([20]), name='bias')


hidden_layer_01 = tf.compat.v1.matmul(x, w1) + b1

# hidden layer
w2 = tf.compat.v1.Variable(tf.zeros([20, 1]), name='weight')
b2 = tf.compat.v1.Variable(tf.zeros([1]), name='bias')

hidden_layer_02 = tf.compat.v1.sigmoid(tf.matmul(hidden_layer_01, w2) + b2)

# output layer
w3 = tf.compat.v1.Variable(tf.zeros([1, 1]), name='weight')
b3 = tf.compat.v1.Variable(tf.zeros([1]), name='bias')


#2. 모델구성                                                       
hypothesis = tf.compat.v1.matmul(hidden_layer_02, w3) + b3


#3-1 컴파일                   
loss = -tf.reduce_mean(y * tf.log(hypothesis) + (1 - y) * tf.log(1 - hypothesis))
optimizer = tf.train.AdamOptimizer(learning_rate=1e-6) 
train = optimizer.minimize(loss)



with tf.compat.v1.Session() as sess : 
    sess.run(tf.global_variables_initializer())
    
    epochs = 5001
    for step in range(epochs):
        cost_val, hy_val, _ = sess.run([loss, hypothesis, train], feed_dict={x:x_data, y:y_data})
        if step %20 == 0:
            print(epochs, 'loss :', cost_val, '\n', hy_val)
   
    y_predict = sess.run(tf.cast(hy_val>=0.5, dtype=tf.float32)) # 이 값이 참이면 1 거짓이면 0
    # y_predict = x1_data * W1_val + x2_data *W2_val + x3_data * W3_val + b_val # r2 : 0.9538024379634447
    
    
    
    print('y예측 :', y_predict)

    r2 = r2_score(y_data, y_predict)
    print('acc :', r2) 

s2