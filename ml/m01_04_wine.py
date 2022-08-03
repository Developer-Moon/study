from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score
from sklearn.datasets import load_wine
import numpy as np 

from sklearn.svm import LinearSVC

import tensorflow as tf            
tf.random.set_seed(66)


#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target 
print(x.shape) # (178, 13)
print(y.shape) # (178,)
# print(np.unique(x))
print(np.unique(y, return_counts=True))     # [0 1 2] - (array([0, 1, 2]), array([59, 71, 48], dtype=int64)) 0이 59개  1이 71개  2가 48개

x_train, x_test, y_train, y_test = train_test_split(x, y,
    test_size=0.33,
    random_state=72
    )



#2. 모델구성
model = LinearSVC()

                           
#3. 컴파일, 훈련
model.fit(x_train, y_train)


#4. 평가, 예측
results = model.score(x_test, y_test)
y_predict = model.predict(x_test)

print('acc : ', results)


# acc스코어 : 0.5932203389830508

# 머신러닝 사용 acc :  0.847457627118644



