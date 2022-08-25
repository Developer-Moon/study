import tensorflow as tf
import numpy as np
tf.set_random_seed(123)

x_data = [[1, 2, 1, 1], # (8, 4)      x = (M, 4),  w = (4, 3),  y = (N, 3),  b = (1, 3)
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 6, 7]]

y_data = [[0, 0, 1],    # 2 (8, 3)
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],    # 1
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],    # 0
          [1, 0, 0]]

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 4])
w = tf.compat.v1.Variable(tf.random_normal([4, 3]), name='weight')
b = tf.compat.v1.Variable(tf.random_normal([1, 3]), name='bias')    # bias 설명좀 부탁드립니다ㅜㅜ
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])



#2. 모델구성
hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)
# model.add(Dense(3, activation='softmax', input_dim=4))



#3-1. 컴파일
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log*(hypothesis), axis=1))
# loss = categorical_crossentropy

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

# loss만 출력