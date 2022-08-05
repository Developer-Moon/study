from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score, accuracy_score
import numpy as np
#--------------------------------------------------------------------------------#
from sklearn.utils import all_estimators
import warnings
import sklearn as sk
print(sk.__version__)             # 0.24.2
warnings.filterwarnings('ignore') # warning 출력X
#--------------------------------------------------------------------------------#


# 1. 데이터
datasets = load_diabetes()
x = datasets['data']
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,shuffle=True, random_state=9)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5 # n_splits=5 5등분
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66 )  

#2. 모델구성
all_Algorithms = all_estimators(type_filter='classifier') # 분류모델 
# all_Algorithms = all_estimators(type_filter='regressor')  # 회귀모델 
# print(all_Algorithms) 전체 모델 보기
print('모델의 갯수 : ', len(all_Algorithms)) # 모델의 갯수 :  41

for (name, algorithms) in all_Algorithms:   # (key, value)
    try:                                    # try 예외처리
        model = algorithms()
        score = cross_val_score(model, x_train, y_train, cv=kfold)
        
        y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
        acc = r2_score(y_test, y_predict)
        print(name, '의 정답률 :', round(np.mean(score), 4))
    except:
        # continue
        print(name, '은 안나온 놈!!!')
         
"""

"""        