from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold
from sklearn.metrics import r2_score, accuracy_score
from sklearn.svm import LinearSVC, SVC
import numpy as np
from sklearn.datasets import load_wine


#1. 데이터
datasets = load_wine()
x = datasets['data']
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=9)
                      
n_splits = 5              
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)
                    
                                      
#2. 모델구성
model = SVC()


#3. 컴파일, 훈련, 평가, 예측
scores = cross_val_score(model, x_train, y_train, cv=kfold)               # cv=5 라면 kfold를 5로 쓴다
y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
acc = accuracy_score(y_test, y_predict)

print('ACC :', scores, '\ncross_val_score :' , round(np.mean(scores), 4)) # 4번째까지 출력 (반올림을 5번째 자리에서)
print(y_predict)
print('cross_val_predict ACC : ', acc)

# ACC : [0.68965517 0.5862069  0.57142857 0.75       0.64285714] 
# cross_val_score : 0.648
# [0 0 0 1 1 0 1 1 1 1 1 0 0 1 1 0 1 1 0 0 0 0 0 0 0 1 1 1 0 1 0 1 1 0 0 1]        
# cross_val_predict ACC :  0.7222222222222222