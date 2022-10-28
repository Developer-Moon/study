from sklearn.metrics import r2_score, accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
import tensorflow as tf
import numpy as np
tf.compat.v1.set_random_seed(123)


#1. 데이터
datasets = load_diabetes()
x_data = datasets.data
y_data = datasets.target

print(x_data.shape, y_data.shape) # (442, 10) (442,)
y_data = y_data.reshape(-1, 1)
print(x_data.shape, y_data.shape) # (442, 10) (442, 1)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.75, random_state=123)

x = tf.placeholder(tf.float32, shape=[None, 10])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([10, 1]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), name='bias')



#2 모델구성
hypothesis = tf.compat.v1.matmul(x, w) + b # 이 연산을 하기 위해 w의 shape을 맞춰준다



#3-1 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-6) 
train = optimizer.minimize(loss)

print('asdasda')
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
    
    y_predict = sess.run(hypothesis, feed_dict={x:x_test, y:y_test})
    r2 = r2_score(y_test, y_predict)
    # accuracy = tf.reduce_mean(tf.cast(tf.equal(y_predict, y), dtype=tf.float32))  
    print('r2 :', r2)
    print('loss :', cost_val)
    '''
    h , c, a = sess.run([hypothesis, y_predict, r2], feed_dict={x:x_test, y:y_test})
    print(f'predict value : {h[0:5]} \n "original value: \n{c[0:5]} \naccuracy: : {a}')  
    '''
    


  