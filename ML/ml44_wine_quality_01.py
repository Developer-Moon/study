import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt # simple linear
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from xgboost import XGBClassifier,XGBRegressor
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC, SVC 
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
     
#1 데이터     
datasets = pd.read_csv('./_data/wine/winequality-white.csv', index_col=None, header=0, sep=';')



x = datasets.drop(['quality', 'density'], axis=1)
y = datasets['quality']

print(x.columns)
# ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
#  'pH', 'sulphates', 'alcohol']


le = LabelEncoder() 
y = le.fit_transform(y)


sns.set(font_scale= 0.8 )
sns.heatmap(data=datasets.corr(), square= True, annot=True, cbar=True) 
plt.show() 

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=66)


n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=99)


#2. 모델구성
parameters = [
    {'n_estimators':[100,200], 'max_depth':[6,8,10,12], 'n_jobs':[-1,2,4]},
    {'max_depth':[6,8,10,12], 'min_samples_leaf':[3,5,7,10], 'n_jobs':[-1,2,4]},
    {'min_samples_leaf':[3,5,7,10], 'min_samples_split':[2,3,5,10], 'n_jobs':[-1,2,4]},
    {'n_estimators':[100,200], 'max_depth':[6,8,10,12], 'min_samples_split':[2,3,5,10]},
    ]                                                                                       
'''
랜덤포레스트의 파라미터들!
parameters = [
    {'n_estimators':[100,200]},
    {'max_depth':[6,8,10,12]},
    {'min_samples_leaf':[3,5,7,10]},
    {'min_samples_split':[2,3,5,10]},
    {'n_jobs':[-1,2,4]}
]
'''       


#2. 모델구성
model = GridSearchCV(RandomForestClassifier(), parameters, cv=kfold, verbose=1, refit=True, n_jobs=-1)


#3. 훈련
model.fit(x_train, y_train)


#4. 평가, 예측
result = model.score(x_test, y_test)
print('model.score :', result)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('accuracy_score :', acc)


# model.score : 0.6824489795918367
# accuracy_score : 0.6824489795918367