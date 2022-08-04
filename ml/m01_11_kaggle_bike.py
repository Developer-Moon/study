from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error          # mean_squared_error : RMSE
import numpy as np                                               
import pandas as pd
from sklearn.svm import LinearSVR


path = './_data/kaggle_bike/'        
train_set = pd.read_csv(path + 'train.csv', index_col=0)   
test_set = pd.read_csv(path + 'test.csv', index_col=0)  

x = train_set.drop(['casual', 'registered', 'count'], axis=1)  
y = train_set['count']   
  
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, shuffle=True, random_state=5)  
    


#2. 모델구성
model = LinearSVR()



#3. 컴파일, 훈련
model.fit(x_train, y_train)



#4. 평가, 예측
results = model.score(x_test, y_test)             
print('r2 : ', results)

# 머신러닝 사용 - r2 :  0.5620467078361082


