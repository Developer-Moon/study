from bitarray import test
from matplotlib.pyplot import axis
import numpy as np
import pandas as pd
from sklearn import datasets
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import time
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

# dataset.describe()
# dataset.info()
# dataset.isnull().sum()
# pandas의 y라벨의 종류가 무엇인지 확인하는 함수 쓸 것
# numpy에서는 np.unique(y, return_counts=True) => pandas에서 동일한 함수 확인     

# 1. 데이터

path = './_data/kaggle_titanic/'
train_set = pd.read_csv(path + 'train.csv', index_col=0)

print(train_set)
print(train_set.columns)                         
print('train_set의 info :: ', train_set.info())
print(train_set.describe())
print(train_set.isnull().sum())     # Age 177, Cabin 687, Embarked 2
print("========================================================")

test_set = pd.read_csv(path + 'test.csv', index_col=0)

# print(test_set) 
# print(test_set.columns) 
# print('test_set의 info :: ', test_set.info())
# # print(test_set.describe())
# print(test_set.feature_name)
# print(test_set.isnull().sum())  # Age 86, Cabin 327  
# print("========================================================")

submission = pd.read_csv(path + 'gender_submission.csv')
print('train.shape, test.shape, submit.shape', 
      train_set.shape, test_set.shape, submission.shape)    # (891, 11) (418, 10) (418, 2)

# 데이터 전처리

# train_set[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# train_set[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# train_set[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# train_set[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)

# Ticket, Cabin[선실], Name 삭제   
train_set = train_set.drop(['Ticket', 'Cabin', 'Name'], axis=1)
test_set = test_set.drop(['Ticket', 'Cabin', 'Name'], axis=1)


# Age NaN값 변환
train_set['Age'] = train_set['Age'].fillna(train_set.Age.dropna().mode()[0])  # NaN값을 최빈값으로 채운다
test_set['Age'] = test_set['Age'].fillna(train_set.Age.dropna().mode()[0])


# Embarked, Sex NaN값 및 Object => int 변환
train_set['Embarked'] = train_set['Embarked'].fillna(train_set.Embarked.dropna().mode()[0])    # Embarked 부분을 지우고 
train_set['Embarked'] = train_set['Embarked'].map({'S':0, 'C':1, 'Q':2}).astype(int)           # 'S':0, 'C':1, 'Q':2 로 매핑한다 int로
test_set['Embarked'] = test_set['Embarked'].fillna(test_set.Embarked.dropna().mode()[0])
test_set['Embarked'] = test_set['Embarked'].map({'S':0, 'C':1, 'Q':2}).astype(int)

train_set['Sex'] = train_set['Sex'].fillna(train_set.Sex.dropna().mode()[0])
train_set['Sex'] = train_set['Sex'].map({'male':0, 'female':1}).astype(int)
test_set['Sex'] = test_set['Sex'].fillna(test_set.Sex.dropna().mode()[0])
test_set['Sex'] = test_set['Sex'].map({'male':0, 'female':1}).astype(int)

print(train_set.shape, test_set.shape)  # (891, 8) (418, 7)
print(train_set.head(5))
print(test_set.head(5))
print(train_set.isnull().sum())  
print(test_set.isnull().sum())  

# x, y 데이터
x = train_set.drop(['Survived'], axis=1)
print(x)
print(x.columns)    # 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'

y = train_set['Survived']
print(y)
print(y.shape)  # (891,)

# One Hot Encoding
# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)
# print(y)
# print(y.shape)  # (891, 2)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.9, shuffle=True, random_state=3
)

# 2. 모델 구성
model = Sequential()
model.add(Dense(100, activation='linear', input_dim=7))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
            #   metrics=['accuracy'],
              metrics=['accuracy'])  

earlyStopping = EarlyStopping(monitor = 'val_loss', patience=100, mode='min', verbose=1, 
                              restore_best_weights=True)

hist = model.fit(x_train, y_train, epochs=100, batch_size=100,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1)


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
print(y_predict)
y_predict = y_predict.flatten()                 
y_predict = np.where(y_predict > 0.5, 1 , 0)   
print(y_predict)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict)) 


acc = accuracy_score(y_test, y_predict)
print("=====================================================================")   
print('acc 스코어 : ', acc)  

# print("===================================================================")   
# print(hist)     
# print("===================================================================")
# print(hist.history)   
print("=====================================================================")
print(hist.history['loss'])
print("=====================================================================")
# print(hist.history['val_loss'])


#그래프로 비교
font_path = 'C:\Windows\Fonts\malgun.ttf'
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
# plt.title('loss & val_loss')    
plt.title('로스값과 검증로스값')    
plt.ylabel('loss')
plt.xlabel('epochs')
# plt.legend(loc='upper right')   # 우측상단에 라벨표시
plt.legend()   # 자동으로 빈 공간에 라벨표시
plt.show()


#5. 데이터 summit
y_summit = model.predict(test_set)
y_summit = y_summit.flatten()                 
y_summit = np.where(y_summit > 0.5, 1 , 0)   
# print(y_summit)
# print(y_summit.shape)

submission = pd.read_csv('./_data/kaggle_titanic/submission.csv')
submission['Survived'] = y_summit
print(submission)
submission.to_csv('./_data/kaggle_titanic/submission1.csv', index=False)

#==================================================================================#
# random_state=3
# loss :  0.39357537031173706
# acc 스코어 :  acc 스코어 :  0.8666666666666667
#==================================================================================#


