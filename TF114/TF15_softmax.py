from sklearn.metrics import accuracy_score, mean_absolute_error
import tensorflow as tf
import numpy as np
tf.set_random_seed(123)


x_data = [[1,2,1,1],  # x = (N, 4)   w = (4, 3)   y = (N, 3)   b = (1, 3)
          [2,1,3,2],
          [3,1,3,4],
          [4,1,5,5],
          [1,7,5,5],
          [1,2,5,6],
          [1,6,6,6],
          [1,7,6,7]]

y_data = [[0,0,1],
          [0,0,1],
          [0,0,1],
          [0,1,0],
          [0,1,0],
          [0,1,0],
          [1,0,0],
          [1,0,0]]

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 4])
w = tf.Variable(tf.random_normal([4, 3]), name='weight')
b = tf.Variable(tf.random_normal([1, 3]), name='bias')
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])


# 2. 모델
hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)


# 3-1. 컴파일
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis), axis=1))
# model.compile(loss='categorical_crossentropy')

train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

# 3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

epochs = 2001
for step in range(epochs):
    _, hy_val, cost_val, b_val = sess.run([train,hypothesis,loss,b], feed_dict={x:x_data, y:y_data})
    if step%20 == 0:
        print(step, cost_val, hy_val)
        
print('최종: ', cost_val, hy_val)

y_pred = sess.run(hypothesis, feed_dict={x:x_data, y:y_data})

acc = accuracy_score(np.argmax(y_data, axis=1), np.argmax(y_pred, axis=1))
print('acc :', acc)

mae = mean_absolute_error(np.argmax(y_data, axis=1), np.argmax(y_pred, axis=1))
print('mae :', mae)

# acc:  0.875
# mae:  0.125












from sklearn.metrics import accuracy_score, mean_absolute_error
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
b = tf.compat.v1.Variable(tf.random_normal([1, 3]), name='bias')   
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])



#2. 모델구성
hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)
# model.add(Dense(3, activation='softmax', input_dim=4))



#3-1. 컴파일
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))

# loss = categorical_crossentropy
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)




with tf.compat.v1.Session() as sess : 
    sess.run(tf.global_variables_initializer())

    epochs = 101
    for step in range(epochs):
        # sess.run(train)
        cost_val, hy_val, _ = sess.run([loss, hypothesis, optimizer], feed_dict={x:x_data, y:y_data})
        # _, : 앞에 반환은 하지 않겠다 하지만 실행은 하겠다 _는 A라고 해도 되고 변수 같은거다 - 필요가 없어서 반환을 하지 않는다
        if step % 20 == 0:
            # print(step, sess.run(loss), sess.run(W), sess.run(b))
            print(epochs, 'loss :', cost_val, '\n', hy_val)



    y_predict = sess.run(hypothesis, feed_dict={x:x_data, y:y_data})  # 이 값이 참이면 1 거짓이면 0
    # y_predict = x1_data * W1_val + x2_data *W2_val + x3_data * W3_val + b_val # r2 : 0.9538024379634447

    y_predict = np.argmax(y_predict, axis=1)
    y_data = np.argmax(y_data, axis=1)

    acc = accuracy_score(y_data, y_predict)
    print('acc :', acc)

    # mae = mean_absolute_error(y_data, hy_val)
    # print('mae :', mae)sd
