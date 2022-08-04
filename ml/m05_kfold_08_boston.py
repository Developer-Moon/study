from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score  # Kfold - cross_val_score검증하기위해 이걸 쓴다
from sklearn.metrics import r2_score, accuracy_score
import numpy as np

import sklearn as sk
print(sk.__version__) # 0.24.2

# 1. 데이터
datasets = load_iris()
x = datasets['data']
y = datasets['target']

# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,shuffle=True, random_state=9)
                    
n_splits =5                # n_splits=5 5등분
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66 ) #         
                    
                    
                      
#2. 모델구성
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron, LogisticRegression # !논리회귀(분류임)!
from sklearn.neighbors import KNeighborsClassifier # 최근접 이웃
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier # 결정트리를 여러개 랜덤으로 뽑아서 앙상블해서 봄

model = SVC()



#3. 컴파일, 훈련, 평가, 예측
scores = cross_val_score(model, x, y, cv=kfold) # cv=5 라면 kfold를 5로 쓴다
print('ACC :', scores, '\n cross_val_score :' , round(np.mean(scores), 4)) # 4번째까지 출력 (반올림을 5번째 자리에서)


# ACC : [0.96666667 0.96666667 1.         0.93333333 0.96666667] 
# cross_val_score : 0.9667