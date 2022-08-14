from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, \
     GridSearchCV  # 격자 탐색, CV: cross validation
import time
     

#1. 데이터
datasets = load_iris()
x = datasets['data']
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=9)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=9)

parameters = [
    {'C':[1,10,100,1000], 'kernel':['linear'], 'degree':[3,4,5]},                                # 4 x 3 = 12
    {'C':[1,10,100], 'kernel':['rbf'], 'gamma':[0.001, 0.0001]},                                 # 3 x 2 = 6
    {'C':[1,10,100,1000], 'kernel':['sigmoid'], 'gamma':[0.01,0.001,0.0001], 'degree':[3,4]}     # 4 x 3 x 2 = 24
]                                                                                                # 총 42회 파라미터를 한다
                 
                      
#2. 모델구성
# model = SVC(C=1, kernel='linear', degree=3)
model = GridSearchCV(SVC(), parameters, cv=kfold, verbose=1, refit=True, n_jobs=-1) # 모델에 wrapper 개념으로 사용한다
                                                           # refit=True면 최적의 파라미터로 훈련,
                                                           #      =False면 해당 파라미터로 훈련하지 않고 마지막 파라미터로 훈련
                                                           # n_jobs: cpu의 갯수를 몇개 사용할것인지


#3. 컴파일, 훈련
model.fit(x_train, y_train)


#4. 평가
y_predict = model.predict(x_test)

print("model.score : ", model.score(x_test, y_test)) # model.score :  0.9666666666666667
print("accuracy_score", accuracy_score(y_test, y_predict)) # accuracy_score 0.9666666666666667

# y_pred_best = model.best_estimator_.__prepare__(x_test)
# print('최적 튠 ACC : ', accuracy_score(y_test, y_pred_best))