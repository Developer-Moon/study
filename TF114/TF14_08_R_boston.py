from sklearn.metrics import r2_score, accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
import tensorflow as tf
tf.compat.v1.set_random_seed(123)


#1. 데이터
datasets = load_boston()
x_data = datasets.data
y_data = datasets.target

print(x_data.shape, y_data.shape) # (506, 13) (506,)
y_data = y_data.reshape(506, 1)
print(x_data.shape, y_data.shape) # (569, 1)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.75, random_state=123)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 13])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([13, 1]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), name='bias')



#2 모델구성
hypothesis = tf.compat.v1.matmul(x, w) + b  



#3-1 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))                      
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-6).minimize(loss) 

with tf.compat.v1.Session() as sess : 
    sess.run(tf.global_variables_initializer())
    
    epochs = 2001
    for step in range(epochs):
        # sess.run(train)
        cost_val, hy_val, _ = sess.run([loss, hypothesis, optimizer], feed_dict={x:x_train, y:y_train})
        # _, : 앞에 반환은 하지 않겠다 하지만 실행은 하겠다 _는 A라고 해도 되고 변수 같은거다 - 필요가 없어서 반환을 하지 않는다
        if step % 20 == 0:
            # print(step, sess.run(loss), sess.run(W), sess.run(b))
            print(epochs, 'loss :', cost_val, '\n', hy_val)
    
    
'''
    y_predict = tf.cast(hy_val>=0.5, dtype=tf.float32) # 이 값이 참이면 1 거짓이면 0
    # y_predict = x1_data * W1_val + x2_data *W2_val + x3_data * W3_val + b_val # r2 : 0.9538024379634447
    
    
    
    print('y예측 :', y_predict)

    acc = tf.reduce_mean(tf.cast(tf.equal(y_predict, y), dtype = tf.float32))
    print('acc :', acc)


    '''