from sklearn.metrics import r2_score, accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import tensorflow as tf
import pandas as pd
import numpy as np
tf.compat.v1.set_random_seed(123)


#1. 데이터
data = load_iris()
x, y = data.data, data.target
print(x.shape, y.shape)                 # (150, 4) (150,)
print(np.unique(y, return_counts=True)) # (array([0, 1, 2]), array([50, 50, 50], dtype=int64))
y = pd.get_dummies(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=123, stratify=y)


#2. 모델
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 4])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])

w = tf.compat.v1.Variable(tf.zeros([4, 50]), name='weight') # 히든레이어를 20개 하려면
b = tf.compat.v1.Variable(tf.zeros([50]), name='bias')      # bias도 20개
hidden_layer = tf.nn.relu(tf.matmul(x, w) + b)

w = tf.compat.v1.Variable(tf.compat.v1.zeros([50,100]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([100]), name='bias')
hidden_layer = tf.matmul(hidden_layer, w) + b

w = tf.compat.v1.Variable(tf.compat.v1.zeros([100,100]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([100]), name='bias')
hidden_layer = tf.matmul(hidden_layer, w) + b

w = tf.compat.v1.Variable(tf.compat.v1.zeros([100,3]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([3]), name='bias')
hypothesis = tf.compat.v1.nn.softmax(tf.matmul(hidden_layer, w) + b)


#3-1 컴파일 
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))
train = tf.train.AdadeltaOptimizer(learning_rate=1e-1).minimize(loss)


#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

epochs = 1001
for step in range(epochs) :
    _, hy_val, cost_val, b_val = sess.run([train, hypothesis, loss, b], feed_dict={x:x_train, y:y_train})
    if step % 20 == 0 :
        print(step, cost_val, hy_val)
        
print('최종 : ', cost_val, hy_val)

_, y_predict = sess.run([train, hypothesis], feed_dict={x:x_test, y:y_test})

y_test = y_test.values

acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_predict, axis=1))
print('acc : ',acc)

mae = mean_absolute_error(np.argmax(y_test, axis=1), np.argmax(y_predict, axis=1))
print('mae : ',mae)
sess.close()

# acc : 0.3157894736842105
# mae : 1.0263157894736843