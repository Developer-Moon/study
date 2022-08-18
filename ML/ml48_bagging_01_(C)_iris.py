from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import load_iris
from xgboost import XGBClassifier, XGBRegressor
from imblearn.over_sampling import SMOTE # pip insatll imblearn
#----------------------------------------------------------------------------------------------------------------#
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, RandomForestRegressor 
from sklearn.linear_model import LogisticRegression # (논리회귀)이진분류
from sklearn.tree import DecisionTreeClassifier
#----------------------------------------------------------------------------------------------------------------#


#1. 데이터
datasets = load_iris()
x, y = datasets.data, datasets.target
print(x.shape, y.shape) # (569, 30) (569,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66, stratify=y)

smote = SMOTE(random_state=123, k_neighbors=4)
x_train, y_train = smote.fit_resample(x_train, y_train)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=9)


#2. 모델구성
parameters = {'n_estimators' : [100, 200, 300, 400, 500, 1000],
               'learning_rate': [0.1, 0.2, 0.3, 0.5, 1, 0.01, 0.001],
               'max_depth': [None, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              'gamma': [0, 1, 2, 3, 4, 5, 7, 10, 100],
             'min_child_weight': [0, 0.01, 0.001, 0.1, 0.5, 1, 5, 10],
            #   'subsample': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1],
            #   'colsample_bytree': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1],
            #   'colsample_bylevel': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1],
            #   'colsample_bynode': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] ,
            #   'reg_alpha': [0, 0.1, 0.01, 0.001, 1, 2, 10],
            #   'reg_lambda':[0, 0.1, 0.01, 0.001, 1, 2, 10]
              }

xgb = XGBClassifier(random_state = 123)
model = GridSearchCV(xgb, parameters, cv=kfold, n_jobs=8)

# model = BaggingClassifier(DecisionTreeClassifier(),   
#                           n_estimators=100,       
#                           n_jobs=-1,
#                           random_state=66,
#                           )

# model = BaggingClassifier(XGBClassifier(n_estimators=100,
#                                         learning_rate=0.1,
#                                         max_depth=2,
#                                         gamma=0,
#                                         min_child_weight=0.01,
#                                         subsample=0,
#                                         colsample_bytree=0.1,
#                                         colsample_bylevel=0.2,
#                                         colsample_bynode=0.3,
#                                         reg_alpha=.001,
#                                         reg_lambda=10))

#3. 훈련
model.fit(x_train, y_train)


#4. 평가, 예측
print(model.score(x_test, y_test)) # 0.9666666666666667
print('최적의 매개변수 :', model.best_estimator_)              # 최적의 매개변수 : SVC(C=10, kernel='linear')
print('최적의 파라미터 :', model.best_params_)                 # 최적의 파라미터 : {'C': 10, 'degree': 3, 'kernel': 'linear'}
print('best_score :', model.best_score_)   
