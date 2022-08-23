# y = wx + b
import tensorflow as tf
tf.set_random_seed(123) # 텐서플로 랜덤시드 고정값


#1. 데이터
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

W = tf.Variable(333, dtype=tf.float32)
b = tf.Variable(77, dtype=tf.float32)
W = tf.Variable(tf.random_normal([1]), dtype=tf.float32) # 숫자는 개수를 의미한다 [0.18394065 -0.04782458 -0.10635024]
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32) # [0.2133712]
# random_normal  : 정규분포로부터의 난수값을 반환한다 - 정규분포는 장기간 축적되어 있는 데이터를 기반으로 다음 데이터를 예상하는 방법
# random_uniform : 균일분포로부터의 난수값을 반환한다 - 균등분포는 앞으로 예상가지 않은 데이터를 말한다

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(W))
print(sess.run(b))


'''
#2. 모델구성
hypothesis = x * W + b



#3-1 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)



#3-2 훈련
with tf.compat.v1.Session() as sess : # sess = tf.compat.v1.Session() 대신 with문을 써서 마지막에 sess close()를 쓰지 않는다
    sess.run(tf.global_variables_initializer())
    epochs = 4001
    for step in range(epochs):
        sess.run(train)
        if step % 20 == 0:
            print(step, sess.run(loss), sess.run(W), sess.run(b))
'''        
