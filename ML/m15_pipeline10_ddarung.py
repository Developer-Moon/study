from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import pandas as pd
#----------------------------------------------------------------------------------------------------------------#
from sklearn.pipeline import make_pipeline
#----------------------------------------------------------------------------------------------------------------#


#1. 데이터
path = './_data/ddarung/'                                       
train_set = pd.read_csv(path + 'train.csv', index_col=0)                                                               
test_set = pd.read_csv(path + 'test.csv', index_col=0) 

train_set = train_set.fillna(train_set.mean())
test_set = test_set.fillna(test_set.mean())   
                                                                                           
train_set = train_set.dropna() 

x = train_set.drop(['count'], axis=1)        
y = train_set['count']                  

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=1234)


#2. 모델구성
# model = make_pipeline(MinMaxScaler(), SVC()) 
model = make_pipeline(MinMaxScaler(), RandomForestRegressor()) 


#3. 훈련
model.fit(x_train, y_train) # 스케일이 제공된 상태, fit과 fit_transform이 같이 돌아간다 - fit_transform을 한 다음 fit을 한다 


#4. 평가, 훈련
result = model.score(x_test, y_test)
print('model.score :', result)

# model.score : 0.7891013801759794