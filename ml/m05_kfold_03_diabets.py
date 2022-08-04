from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, KFold, cross_val_score  # Kfold - cross_val_score검증하기위해 이걸 쓴다
from sklearn.metrics import r2_score, accuracy_score
from sklearn.svm import LinearSVC, SVC, SVR
import numpy as np


# 1. 데이터
datasets = load_diabetes()
x = datasets['data']
y = datasets['target']

# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,shuffle=True, random_state=9)
                    
n_splits = 5 # n_splits=5 5등분
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)      
                    
                    
                      
#2. 모델구성
model = SVR()



#3. 컴파일, 훈련, 평가, 예측
scores = cross_val_score(model, x, y, cv=kfold) # cv=5 라면 kfold를 5로 쓴다
print('ACC :', scores, '\ncross_val_score :' , round(np.mean(scores), 4)) # 4번째까지 출력 (반올림을 5번째 자리에서)

# ACC : [0.89473684 0.92982456 0.89473684 0.92105263 0.96460177]
# cross_val_score : 0.921