from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error          # mean_squared_error : RMSE
import numpy as np                                               
import pandas as pd

from sklearn.svm import LinearSVR


#1. 데이타 
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
model = LinearSVR()



#3. 컴파일, 훈련
model.fit(x_train, y_train)



#4. 평가, 예측
results = model.score(x_test, y_test)    
y_predict = model.predict(x_test)                          # 예측해서 나온 값

def RMSE(y_test, y_predict):                               # 이 함수는 y_test, y_predict를 받아 들인다
    return np.sqrt(mean_squared_error(y_test, y_predict))  # 내가 받아들인 y_test, y_predict를 mean_squared_error에 넣는다 그리고 루트를 씌운다 그리고 리턴
                                                           # mse가 제곱하여 숫자가 커져서 (sqrt)루트를 씌우겠다 
rmse = RMSE(y_test, y_predict)  
print("RMSE :", rmse)           
print('r2 : ', results)

# 머신러닝 사용 
# RMSE : 54.757326870935614
# r2 :  0.5620467078361082




# y_summit = model.predict(test_set)

# print(y_summit)
# print(y_summit.shape) # (715, 1)

# submission = pd.read_csv('./_data/ddarung/submission.csv')
# submission['count'] = y_summit
# print(submission)
# submission.to_csv('./_data/ddarung/submission2.csv', index = False)


# loss : 3058.065673828125
# RMSE : 55.299774499917866