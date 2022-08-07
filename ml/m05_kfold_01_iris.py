from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold # Kfold - cross_val_score검증하기위해 이걸 쓴다
from sklearn.metrics import r2_score, accuracy_score
from sklearn.svm import LinearSVC, SVC
import numpy as np
from sklearn.datasets import load_iris


# 1. 데이터
datasets = load_iris()
x = datasets['data']
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=9)
                    
n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66 )      
                    
                        
#2. 모델구성
model = SVC()


#3. 컴파일, 훈련, 평가, 예측
scores = cross_val_score(model, x_train, y_train, cv=kfold) # cv=5 라면 kfold를 5로 쓴다
print('ACC :', scores, '\ncross_val_score :' , round(np.mean(scores), 4)) # 4번째까지 출력 (반올림을 5번째 자리에서)

y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
print(y_predict)

acc = accuracy_score(y_test, y_predict)
print('cross_val_predict ACC : ', acc)

# ACC : [0.96666667 0.96666667 1.         0.93333333 0.96666667] 
# cross_val_score : 0.9667
# [2 1 2 2 1 0 0 0 1 0 0 1 1 1 0 1 0 1 2 0 0 0 2 0 2 1 0 2 0 2]
# cross_val_predict ACC :  1.0