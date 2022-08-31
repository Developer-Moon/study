from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import tensorflow as tf
import numpy as np
tf.compat.v1.set_random_seed(123)


#1. 데이터
datasets = load_breast_cancer()
x_data = datasets.data
y_data = datasets.target

print(x_data.shape) # (569, 30)
print(y_data.shape) # (569,
print(np.unique(y_data, return_counts=True)) # (array([0, 1]), array([212, 357], dtype=int64))
y_data = y_data.reshape(-1, 1)
print(y_data.shape) # (569, 1)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.75, random_state=123, stratify=y_data)

x = tf.placeholder(tf.float32, shape=[None, 30])
y = tf.placeholder(tf.float32, shape=[None, 1])

# hidden layer
w1 = tf.compat.v1.Variable(tf.random_normal([30, 50]), name='weight') # 히든레이어를 20개 하려면
b1 = tf.compat.v1.Variable(tf.random_normal([50]), name='bias')      # bias도 20개

hidden_layer_01 = tf.compat.v1.matmul(x, w1) + b1

# hidden layer
w2 = tf.compat.v1.Variable(tf.random_normal([50, 10]), name='weight') # 두번째 층이 받아 들이는 것
b2 = tf.compat.v1.Variable(tf.random_normal([10]), name='bias')

hidden_layer_02 = tf.compat.v1.matmul(hidden_layer_01, w2) + b2

# output layer
w3 = tf.compat.v1.Variable(tf.random_normal([10, 1]), name='weight') # 두번째 층이 받아 들이는 것
b3 = tf.compat.v1.Variable(tf.random_normal([1]), name='bias')


#2. 모델구성
hypothesis = tf.compat.v1.nn.softmax(tf.matmul(hidden_layer_02, w3) + b3)          
                                        


#3-1 컴파일 
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))
optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.01) 
train = optimizer.minimize(loss)



with tf.compat.v1.Session() as sess : 
    sess.run(tf.global_variables_initializer())
    
    epochs = 1001
    for step in range(epochs):
        cost_val, hy_val, _ = sess.run([loss, hypothesis, train], feed_dict={x:x_data, y:y_data})
        if step %20 == 0:
            print(epochs, 'loss :', cost_val, '\n', hy_val) 
    
    y_predict = sess.run(hypothesis, feed_dict={x:x_data, y:y_data}) 
    print('y예측 :', y_predict)
    
    y_predict = np.argmax(y_predict, axis=1)
    y_data = np.argmax(y_data, axis=1)

    acc = accuracy_score(y_data, y_predict)
    print('acc :', acc)     
    
    
    

# acc : 0.6274165202108963
