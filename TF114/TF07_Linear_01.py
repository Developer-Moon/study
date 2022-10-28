# y = wx + b
import tensorflow as tf
tf.set_random_seed(123) # 텐서플로 랜덤시드 고정값


#1. 데이터
x = [1, 2, 3]
y = [1, 2, 3]

W = tf.Variable(1, dtype=tf.float32) # 초기값 1로 지정
b = tf.Variable(1, dtype=tf.float32) # 초기값 1로 지정



#2. 모델구성
hypothesis = x * W + b               # 실제 연산 방법 - hypothesis : 가설
# 행렬 연산이기 때문에 x와 w순서가 중요하다 즉 인풋값(x)에 웨이트(w)를 곱한다



#3-1 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y)) # loss는 mse라는걸 풀어서 정의
# 1. 예측값(hypothesis)에서 원래값(y)을 뺀다(거리를 구한다)
# 2. 제곱(square)
# 3. 평균을 구한다(reduce_mean)


optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01) # 러닝레이트는 내려가는 간격(그래프에서)을 말 한다 # 경사하강법()
                                                                  # 그래프에서 y는 loss, x는 에포
train = optimizer.minimize(loss) # 로스값의 최소값을 리턴
# 텐서2 - model.compile(loss='mse', optimizer='sgd') 


#3-2 훈련
sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer()) # 첫번째 sess 변수들을 초기화 해야한다

epochs = 2001
for step in range(epochs): # epochs 2001번
    sess.run(train)        # model.fit
    if step %20 == 0:      # 20번에 1번만 출력 verbose 조절, 훈련수를 20으로 나눴을때 0이면 출력한다 
        print(step, sess.run(loss), sess.run(W), sess.run(b))
        
sess.close()               # 끝날때 마다 세션을 닫아줘야한다.

