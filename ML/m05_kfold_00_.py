import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold # Kfold - cross_val_score 검증하기위해 이걸 쓴다
from sklearn.svm import SVC
from sklearn.datasets import load_iris


#1. 데이터
datasets = load_iris()
x = datasets['data']
y = datasets['target']

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=99)
             
                      
#2. 모델구성
model = SVC()
    
    
#3. 컴파일, 훈련, 평가, 예측
score = cross_val_score(model, x, y, cv=kfold)
# score = cross_val_score(model, x, y, cv=5)
# 이렇게 하면 위에서 kFold로 따로 정의하지 않아도 된다 대신 파라미터들을 건들 수 있는게 줄어든다

print('acc :', score, '\ncross_val_score :', round(np.mean(score), 4)) # 4번째까지 출력 (반올림을 5번째 자리에서)

# acc : [0.96666667 1.         0.96666667 0.93333333 0.96666667]
# cross_val_score : 0.9667