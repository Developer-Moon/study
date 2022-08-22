# y = wx + b
import tensorflow as tf
tf.set_random_seed(123) # 텐서플로 랜덤시드 고정값


#1. 데이터
x = [1, 2, 3, 4, 5]
y = [1, 2, 3, 4, 5]

W = tf.Variable(333, dtype=tf.float32)
b = tf.Variable(77, dtype=tf.float32)



#2. 모델구성
hypothesis = x * W + b



#3-1 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)



#3-2 훈련
sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

epochs = 4001
for step in range(epochs):
    sess.run(train)
    if step %20 == 0:
        print(step, sess.run(loss), sess.run(W), sess.run(b))
        
sess.close()


