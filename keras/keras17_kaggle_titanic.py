# datasets.describe()
# datasets.info()
# datasets.insull().sum()

# pandas의 y라벨의 종류가 무엇인지 확인하는 함수 쓸 것 
# numpy에서는 np.unique(y, return_counts=True)

import datetime as dt
import numpy as np                                               
import pandas as pd 

from tensorflow.python.keras.models import Sequential   
from tensorflow.python.keras.layers import Dense 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error    


#plt 폰트 깨짐 현상 #
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/malgun.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
#plt 폰트 깨짐 현상 #

import tensorflow as tf            
tf.random.set_seed(66)         #텐서플로의 난수표 66번 사용 이 난수는 w = 웨이트



#1. 데이타 
path = './_data/kaggle_titanic/'      
                                  
train_set = pd.read_csv(path + 'train.csv', index_col=0)                       
print(train_set.shape)       # (891, 11) 

test_set = pd.read_csv(path + 'test.csv', index_col=0)                                     
print(test_set.shape)        # (418, 10)


print(train_set.columns)     # ['Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'] - Survived
print(test_set.columns)      # ['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']  

print(train_set.info())      # non-null count : 결측치 
print(train_set.describe())  

print(train_set.shape)       # (891, 11)

#####결측치 처리 1. 제거 ######
print(train_set.isnull().sum())
train_set = train_set.dropna()
test_set = test_set.fillna(test_set.mean())
print(train_set.isnull().sum())
print(train_set.shape)  # (183, 11)


x = train_set.drop(['Survived'], axis=1)   
print(x)
print(x.columns)
print(x.shape)    # (183, 10)

y = train_set['Survived']   #카운트 컬럼만 빼서 y출력
print(y)  
print(y.shape)    # (183,)

x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.9,
    shuffle=True, 
    random_state=3
    )


#2. 모델구성
model = Sequential()
model.add(Dense(500, input_dim=9))  
model.add(Dense(110))
model.add(Dense(120))
model.add(Dense(130))
model.add(Dense(140))
model.add(Dense(150))
model.add(Dense(140))

model.add(Dense(1))








