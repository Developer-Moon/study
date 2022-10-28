from sklearn.metrics import r2_score, accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import tensorflow as tf
import numpy as np
tf.compat.v1.set_random_seed(123)


#1. 데이터
datasets = load_breast_cancer()
x_data = datasets.data
y_data = datasets.target

print(y_data.shape) # (569,)
print(np.unique(y_data, return_counts=True)) # (array([0, 1]), array([212, 357], dtype=int64))
y_data = y_data.reshape(-1, 1)
print(y_data.shape) # (569, 1)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.75, random_state=123, stratify=y_data)

x = tf.placeholder(tf.float32, shape=[None, 30])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.zeros([30, 1]), name='weight')
b = tf.compat.v1.Variable(tf.zeros([1]), name='bias')



#2 모델구성
hypothesis = tf.compat.v1.sigmoid(tf.matmul(x, w) + b ) # sigmoid를 씌운다
                                                        # model.add(Dense)1, activation='sigmoid', input_dim=2)) 와 같다


#3-1 컴파일
# loss = tf.reduce_mean(tf.square(hypothesis - y))                      
loss = -tf.reduce_mean(y * tf.log(hypothesis) + (1 - y) * tf.log(1 - hypothesis)) # binary_crossentropy 


# optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-3) 
optimizer = tf.train.AdamOptimizer(learning_rate=0.00000117) 
train = optimizer.minimize(loss)

y_predict = tf.cast(hypothesis>=0.5, dtype=tf.float32)
y_test = tf.cast(hypothesis>=0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(y_predict, y), dtype=tf.float32))  

print('hi')
with tf.compat.v1.Session() as sess : 
    sess.run(tf.global_variables_initializer())
    
    epochs = 2001
    for step in range(epochs):
        # sess.run(train)
        cost_val, hy_val, _ = sess.run([loss, hypothesis, train], feed_dict={x:x_train, y:y_train})
        # _, : 앞에 반환은 하지 않겠다 하지만 실행은 하겠다 _는 A라고 해도 되고 변수 같은거다 - 필요가 없어서 반환을 하지 않는다
        if step % 20 == 0:
            # print(step, sess.run(loss), sess.run(W), sess.run(b))
            print(epochs, 'loss :', cost_val, '\n', hy_val)
    
    h , c, a = sess.run([hypothesis, y_predict, accuracy], feed_dict={x:x_test, y:y_test})
    print(f'predict value : {h[0:5]} \n "original value: \n{c[0:5]} \naccuracy: : {a}')  
     
    acc = accuracy_score(y_test, y_predict)
    print('acc :', acc)


  