from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold # Kfold - cross_val_score검증하기위해 이걸 쓴다
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, accuracy_score
from sklearn.svm import SVR
import numpy as np
import pandas as pd


#1. 데이터 
path = './_data/ddarung/'                                       
train_set = pd.read_csv(path + 'train.csv', index_col=0)                                                               
test_set = pd.read_csv(path + 'test.csv', index_col=0) 

train_set = train_set.fillna(train_set.mean())
test_set = test_set.fillna(test_set.mean())   
                                                                                           
train_set = train_set.dropna() 

x = train_set.drop(['count'], axis=1)        
y = train_set['count']                  

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, shuffle=True, random_state=16)

scaler = RobustScaler()
scaler.fit(x_train)
scaler.fit(test_set)
test_set = scaler.transform(test_set)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
                      
n_splits =5  
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66 )    
                    
                                 
#2. 모델구성
model = SVR()


#3. 컴파일, 훈련, 평가, 예측
scores = cross_val_score(model, x_train, y_train, cv=kfold) # cv=5 라면 kfold를 5로 쓴다
print('ACC :', scores, '\ncross_val_score :' , round(np.mean(scores), 4)) # 4번째까지 출력 (반올림을 5번째 자리에서)

y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
# print(y_predict)

r2 = r2_score(y_test, y_predict)
print('cross_val_predict r2 : ', r2)

# ACC : [0.39336819 0.35894708 0.39073978 0.33232426 0.36778279] 
# cross_val_score : 0.3686
# cross_val_predict r2 :  0.2448154014776499