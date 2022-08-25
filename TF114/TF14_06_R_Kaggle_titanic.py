from sklearn.metrics import r2_score, accuracy_score, mean_absolute_error
import tensorflow as tf
tf.compat.v1.set_random_seed(123)


#1. 데이터
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]] # (6, 2)
y_data = [[0], [0], [0], [1], [1], [1]]                   # (6, 1)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([2, 1]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), name='bias')



#2 모델구성
hypothesis = tf.compat.v1.sigmoid(tf.matmul(x, w) + b ) # sigmoid를 씌운다
                                                        # model.add(Dense)1, activation='sigmoid', input_dim=2)) 와 같다


#3-1 컴파일
# loss = tf.reduce_mean(tf.square(hypothesis - y))                      
loss = -tf.reduce_mean(y * tf.log(hypothesis) + (1 - y) * tf.log(1 - hypothesis)) # binary_crossentropy 


optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-3) 
train = optimizer.minimize(loss)



with tf.compat.v1.Session() as sess : 
    sess.run(tf.global_variables_initializer())
    
    epochs = 1001
    for step in range(epochs):
        # sess.run(train)
        cost_val, hy_val, _ = sess.run([loss, hypothesis, train], feed_dict={x:x_data, y:y_data})
        # _, : 앞에 반환은 하지 않겠다 하지만 실행은 하겠다 _는 A라고 해도 되고 변수 같은거다 - 필요가 없어서 반환을 하지 않는다
        if step %20 == 0:
            # print(step, sess.run(loss), sess.run(W), sess.run(b))
            print(epochs, 'loss :', cost_val, '\n', hy_val)
    
   
    y_predict = sess.run(tf.cast(hy_val>=0.5, dtype=tf.float32)) # 이 값이 참이면 1 거짓이면 0
    # y_predict = x1_data * W1_val + x2_data *W2_val + x3_data * W3_val + b_val # r2 : 0.9538024379634447
    
    
    
    print('y예측 :', y_predict)

    acc = accuracy_score(y_data, y_predict)
    print('acc :', acc)

    mae = mean_absolute_error(y_data, hy_val)
    print('mae :', mae)
    
