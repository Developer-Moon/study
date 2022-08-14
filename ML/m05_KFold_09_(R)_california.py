from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold
from sklearn.metrics import r2_score, accuracy_score
from sklearn.svm import SVR
import numpy as np
from sklearn.datasets import fetch_california_housing


#1. 데이터
datasets = fetch_california_housing()
x = datasets['data']
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=9)
                      
n_splits = 5              
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)
                    
                                      
#2. 모델구성
model = SVR()


#3. 컴파일, 훈련, 평가, 예측
scores = cross_val_score(model, x_train, y_train, cv=kfold)               # cv=5 라면 kfold를 5로 쓴다
y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
r2 = r2_score(y_test, y_predict)

print('r2 :', scores, '\ncross_val_score :' , round(np.mean(scores), 4)) # 4번째까지 출력 (반올림을 5번째 자리에서)
print(y_predict)
print('cross_val_predict r2 : ', r2)

# r2 : [-0.045286   -0.02637135 -0.00948287 -0.0323298  -0.03411621] 
# cross_val_score : -0.0295
# [1.77163891 1.79201214 1.81360294 ... 1.75352998 1.80043611 1.75829284]
# cross_val_predict r2 :  -0.050740122512172636
