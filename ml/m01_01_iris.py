
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC # support vector classifier - 레거시안 사이킷런 모델 
import tensorflow as tf
import numpy as np
tf.random.set_seed(66)


# 1. 데이터
datasets = load_iris()
print(datasets.DESCR) 
print(datasets.feature_names)

x = datasets['data']
y = datasets['target']
print(x, '\n', y)
print(x.shape) # (150, 4)
print(y.shape) # (150,)
print('y의 라벨값: ', np.unique(y))

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,shuffle=True, random_state=9)


      
#2. 모델구성
model = LinearSVC() 




#3. 컴파일, 훈련
model.fit(x_train, y_train) # 원핫이 필요없다 훈련은 통상 100번 하는데 찾아봐라 여기에 컴파일이 포함되어있다



#4. 평가, 예측
results = model.score(x_test, y_test) # evaluate대신 score    # 회귀에서는 r2_score 분류에서는 accuracy - auto 설정이라 이해??

print('결과 acc :', results)

y_predict = model.predict(x_test)
# y_predict = tf.argmax(y_predict, axis=1) # 행끼리 비교해서 몇번째 인덱스가 제일 큰지 알려줌
# y_test = tf.argmax(y_test, axis=1) # y_test도 argmax를 해서 같은 리스트를 비교하기

# print(y_test)
# print(y_predict)


# 딥러닝 = 히든레이어가 많아서 시간이 오래 걸림
# 머신러닝 단층 리니어 

# error : 문제가 있어서 중단됨
# bug : 작동은 되지만 문제가 있음

# loss :  0.004345672205090523
# accuracy :  1.0



# 머신러닝 사용
# 결과 acc : 1.0

