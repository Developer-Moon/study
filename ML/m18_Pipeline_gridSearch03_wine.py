from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV, HalvingRandomSearchCV, KFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
#----------------------------------------------------------------------------------------------------------------#
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.decomposition import PCA # 컬럼을 압축한다
#----------------------------------------------------------------------------------------------------------------#


#1. 데이터
datasets = load_wine()
x = datasets.data 
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1234)

parameters = [
    {'RF__n_estimators' : [100, 200], 'RF__max_depth' : [6, 8, 10, 12, 14, 16]},                                  
    {'RF__max_depth' : [6, 8, 10, 12], 'RF__min_samples_leaf' : [3, 5, 7, 10, 13]},
    {'RF__min_samples_leaf' : [3, 5, 7, 10], 'RF__min_samples_split' : [2, 3, 5, 10, 15, 20]}
]

n_splits=5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=66)


#2. 모델구성
pipe = Pipeline([('minmax', MinMaxScaler()), ('hi', PCA()), ('RF', RandomForestClassifier())]) # 변수들을 parameters에 명시해줘야 한다
          

#3. 훈련
model = GridSearchCV(pipe, parameters, cv=kfold, verbose=1)
model.fit(x_train, y_train)


#4. 평가, 훈련
result = model.score(x_test, y_test)
print('model.score :', result)

from sklearn.metrics import accuracy_score 
y_predict = model.predict(x_test) # predict는 make_pipeline을 이용하여 scaler가 적용된 상태다
acc = accuracy_score(y_test, y_predict)
print('accuracy :', acc)

# normal Scaler - acc : 0.9666666666666667
# make_pipeline - model.score : 1.0

# GridSearchCV
# - model.score : 0.9638888888888889
# - accuracy : 0.9638888888888889
