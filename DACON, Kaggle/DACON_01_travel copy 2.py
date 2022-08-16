import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier,XGBRegressor
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, accuracy_score

### 1.데이터 ###
path  = './_data/dacon_travel/' 

train = pd.read_csv( path + 'train.csv', index_col=0)
test = pd.read_csv(path + 'test.csv', index_col=0)
submit = pd.read_csv(path + 'sample_submission.csv',index_col=0)

print(train.shape, test.shape) # (1955, 19) (2933, 18)


##################### 라벨인코더 ######################
le = LabelEncoder()

idxarr = train.columns
idxarr = np.array(idxarr)

for i in idxarr:
      if train[i].dtype == 'object':
        train[i] = le.fit_transform(train[i])
        test[i] = le.fit_transform(test[i])


### 상관관계 ###
# sns.set(font_scale= 0.8 )
# sns.heatmap(data=train.corr(), square= True, annot=True, cbar=True) # square: 정사각형, annot: 안에 수치들 ,cbar: 옆에 bar

# plt.show() 
# train.to_csv(path + 'train22.csv',index=False)

### 결측치 ###
#트레인,테스트 합치기 #
alldata = pd.concat((train, test), axis=0)
alldata_index = alldata.index

alldata = alldata.drop(['Age','MonthlyIncome'],axis=1)

print(alldata.shape)
train = alldata[:len(train)]
test = alldata[len(train):]

train = train.dropna()


mean = test['DurationOfPitch'].mean()
test['DurationOfPitch'] = test['DurationOfPitch'].fillna(mean)

mean = test['NumberOfFollowups'].mean()
test['NumberOfFollowups'] = test['NumberOfFollowups'].fillna(mean)

mean = test['PreferredPropertyStar'].median()
test['PreferredPropertyStar'] = test['PreferredPropertyStar'].fillna(mean)

mean = test['NumberOfTrips'].median()
test['NumberOfTrips'] = test['NumberOfTrips'].fillna(mean)

mean = test['NumberOfChildrenVisiting'].median()
test['NumberOfChildrenVisiting'] = test['NumberOfChildrenVisiting'].fillna(mean)


print(test.isnull().sum())


x = train.drop('ProdTaken',axis=1)
print(x.shape)      #(1955, 18)
y = train['ProdTaken']
print(y.shape)      #(1955,)
print(submit.shape) #(2933, 1)
print(test.columns)

test = test.drop('ProdTaken',axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, 
                                                     random_state=123)


parameters = {'n_estimators' : [100, 200, 300, 400, 500, 1000],
              'learning_rate': [0.1, 0.2, 0.3, 0.5, 1, 0.01, 0.001],
              'max_depth': [None, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              'gamma': [0, 1, 2, 3, 4, 5, 7, 10, 100],
              'min_child_weight': [0, 0.01, 0.001, 0.1, 0.5, 1, 5, 10],
              'subsample': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1],
              'colsample_bytree': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1],
              'colsample_bylevel': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1],
              'colsample_bynode': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] ,
              'reg_alpha': [0, 0.1, 0.01, 0.001, 1, 2, 10],
              'reg_lambda':[0, 0.1, 0.01, 0.001, 1, 2, 10]
              }

#2. 모델 
from sklearn.svm import LinearSVC,SVC
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.pipeline import Pipeline, make_pipeline 
from xgboost import XGBRegressor

model = XGBClassifier(random_state = 66)
model.fit(x_train,y_train)

result = model.score(x_test, y_test)
print('model.score :', result)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('accuracy_score :', acc)
