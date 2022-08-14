from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold
from sklearn.metrics import r2_score, accuracy_score
from sklearn.svm import LinearSVC, SVC
import numpy as np
from sklearn.datasets import load_digits


#1. 데이터
datasets = load_digits()
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

# ACC : [0.98958333 0.97222222 0.98954704 0.97909408 0.98954704] 
# cross_val_score : 0.984
# [1 1 7 2 4 0 1 8 8 3 1 0 5 3 6 2 1 8 2 5 3 9 0 0 6 8 3 2 3 8 0 1 3 2 8 0 1       
#  7 1 3 9 2 1 4 1 1 2 8 4 4 0 2 8 4 8 5 7 3 8 8 9 2 4 1 5 2 0 5 1 4 8 4 7 6       
#  1 9 5 1 7 6 4 0 2 5 9 1 9 7 8 7 6 4 1 5 3 4 8 8 8 6 7 9 4 1 6 4 0 5 7 8 1       
#  3 4 3 1 3 8 6 1 5 0 7 8 9 0 1 9 7 5 6 7 9 9 2 4 3 8 9 5 5 2 2 1 5 1 0 1 8       
#  5 5 4 5 2 5 1 7 5 5 7 4 9 3 5 4 6 9 0 3 4 1 6 0 6 3 2 8 3 9 2 2 2 8 3 4 2       
#  2 8 3 7 9 2 8 5 0 1 8 9 0 7 5 1 6 9 0 7 5 1 3 7 3 0 9 2 9 9 8 9 4 0 7 8 3       
#  5 3 4 6 6 5 0 9 6 0 6 9 4 1 5 5 0 4 2 2 2 3 4 0 8 0 9 4 5 1 4 1 3 8 4 9 2       
#  8 2 2 7 1 8 2 0 2 9 6 2 9 3 7 4 5 7 4 9 5 6 4 5 9 2 9 1 6 7 9 5 2 0 5 6 1       
#  8 1 3 4 9 5 8 1 2 1 2 6 4 4 9 9 2 3 5 4 2 7 6 3 7 1 6 4 8 0 2 8 4 6 1 7 3       
#  0 0 9 1 9 2 1 9 8 2 6 6 4 6 2 7 0 4 6 5 5 7 8 3 3 8 3]
# cross_val_predict ACC :  0.9666666666666667