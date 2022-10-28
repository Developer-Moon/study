#[실습]#
import numpy as np 
import pandas as pd
from sklearn import metrics
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import Dense,Dropout,Input,LSTM,Conv1D,Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,accuracy_score
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler
import tensorflow as tf
import datetime

#1.데이터
path = './_data/kaggle_titanic/'
train_set = pd.read_csv(path + 'train.csv',index_col =0)
test_set = pd.read_csv(path + 'test.csv', index_col=0)

##########전처리############
train_test_data = [train_set, test_set]
sex_mapping = {"male":0, "female":1}
for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)

print(dataset)

for dataset in train_test_data:
    # 가족수 = 형제자매 + 부모님 + 자녀 + 본인
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    dataset['IsAlone'] = 1
    
    # 가족수 > 1이면 동승자 있음
    dataset.loc[dataset['FamilySize'] > 1, 'IsAlone'] = 0

for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
embarked_mapping = {'S':0, 'C':1, 'Q':2}
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)

for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract('([\w]+)\.', expand=False)
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].apply(lambda x: 0 if x=="Mr" else 1 if x=="Miss" else 2 if x=="Mrs" else 3 if x=="Master" else 4)

train_set['Cabin'] = train_set['Cabin'].str[:1]
for dataset in train_test_data:
    dataset['Age'].fillna(dataset.groupby("Title")["Age"].transform("median"), inplace=True)
for dataset in train_test_data:
    dataset['Agebin'] = pd.cut(dataset['Age'], 5, labels=[0,1,2,3,4])
for dataset in train_test_data:
    dataset["Fare"].fillna(dataset.groupby("Pclass")["Fare"].transform("median"), inplace=True)
for dataset in train_test_data:
    dataset['Farebin'] = pd.qcut(dataset['Fare'], 4, labels=[0,1,2,3])
    drop_column = ['Name', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin']

for dataset in train_test_data:
    dataset = dataset.drop(drop_column, axis=1, inplace=True)
print(train_set.head())


x = train_set.drop(['Survived'], axis=1,)
y = train_set['Survived']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size= 0.3,random_state=61)

# minmax , standard
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_set = scaler.transform(test_set)        
                                                    
print(x_train.shape,x_test.shape)
x_train = x_train.reshape(623,8,1)
x_test = x_test.reshape(268,8,1)


#2.모델구성
model = Sequential()
model.add(Conv1D(200,2,activation='relu', input_shape=(8,1))) 
model.add(Flatten())
model.add(Dense(100,activation= 'relu'))
model.add(Dense(100,activation= 'relu'))
model.add(Dense(100,activation= 'relu'))
model.add(Dense(100,activation= 'relu'))
model.add(Dense(1,activation='sigmoid'))


date= datetime.datetime.now()
date=date.strftime('%m%d_%H%M')

#.컴파일,훈련
model.compile(loss= 'binary_crossentropy',optimizer='adam')
earlyStopping= EarlyStopping(monitor= 'val_loss',patience=10,mode='min',restore_best_weights=True,verbose=1)

# filepath='./_ModelCheckpoint/k24/'
# filename='{epoch:04d}-{val_loss:.4f}.hdf5'

# mcp = ModelCheckpoint(monitor = 'val_loss',mode = 'auto', save_best_only=True, verbose=1,
#                       filepath="".join([filepath,'titanic',date,'_',filename]))

model.fit(x_train, y_train, epochs=200, batch_size=64,validation_split=0.2,callbacks=[earlyStopping], verbose=1)

#4.평가,예측
loss = model.evaluate(x_test,y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

y_predict = y_predict.round() 

acc = accuracy_score(y_test,y_predict)
print('acc : ',acc)

#DNN
# loss :  0.4440535306930542
# acc :  0.8097014925373134

#LSTM
# loss :  0.4711759388446808
# acc :  0.8059701492537313

#Conv1D
# loss :  0.46752122044563293
# acc :  0.7798507462686567