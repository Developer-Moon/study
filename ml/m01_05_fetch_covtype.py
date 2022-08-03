from sklearn.model_selection import train_test_split        
from sklearn.datasets import fetch_covtype                            
from sklearn.metrics import r2_score
import numpy as np 
import pandas as pd 

from sklearn.svm import LinearSVC


import tensorflow as tf            
tf.random.set_seed(66)         #텐서플로의 난수표 66번 사용 이 난수는 w = 웨이트

#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target 
print(x.shape, y.shape) # (581012, 54) (581012,)
print(np.unique(y, return_counts=True))     # (array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510], dtype=int64))
                                            # 4가 잘 안나온다? 4를 삭제할까? 라는 생각을 해야한다??
                                            # 이 데이터를 증폭시킨다?                     
                                            # !!!!!!!!!!!!!!!!! - 여기서 7개의 컬럼인데 


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



###################################
# loss :  1.1296087503433228
# accuracy :  0.5815765857696533
# acc스코어 : 0.5815765591913797
###################################

# 머신러닝 사용 acc :  0.5494069909353584