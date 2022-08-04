from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score, accuracy_score
import pandas as pd
#--------------------------------------------------------------------------------#
from sklearn.utils import all_estimators
import warnings
import sklearn as sk
print(sk.__version__)             # 0.24.2
warnings.filterwarnings('ignore') # warning 출력X
#--------------------------------------------------------------------------------#


#1. 데이터
path = './_data/ddarung/'                                        
train_set = pd.read_csv(path + 'train.csv', index_col=0)          
test_set = pd.read_csv(path + 'test.csv', index_col=0)            

train_set = train_set.fillna(train_set.mean())
test_set = test_set.fillna(test_set.mean())   

print(train_set.isnull().sum())                                            
print(test_set.isnull().sum())                   
             
train_set = train_set.dropna()               

x = train_set.drop(['count'], axis=1)        
print(x)
print(x.columns)                             
print(x.shape)                              

y = train_set['count']                        
print(y)  
print(y.shape)                                  

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, shuffle=True, random_state=16)



#2. 모델구성
# all_Algorithms = all_estimators(type_filter='classifier') # 분류모델
all_Algorithms = all_estimators(type_filter='regressor')  # 회귀모델
# print(all_Algorithms) 전체 모델 보기
print('모델의 갯수 :', len(all_Algorithms)) # 모델의 갯수 :  41

for (name, algorithms) in all_Algorithms:   # (key, value)
    try:                                    # try 예외처리
        model = algorithms()
        model.fit(x_train, y_train)
        
        y_predict = model.predict(x_test)
        acc = r2_score(y_test, y_predict)
        print(name, '의 정답률 :', acc)
    except:
        # continue
        print(name, '은 안나온 놈!!!')
        

"""

"""        