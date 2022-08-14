from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold, StratifiedKFold
from sklearn.metrics import r2_score, accuracy_score
from sklearn.svm import LinearSVC, SVC
import numpy as np
from sklearn.datasets import load_iris

# kfold와 StratifiedKFold비교

#1. 데이터
datasets = load_iris()
x = datasets['data']
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,shuffle=True, random_state=9)

n_splits = 5
# kfold = KFold(n_splits=n_splits, shuffle=True, random_state=99)
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=99)


#2. 모델구성
model = SVC()


#3. 컴파일, 훈련, 평가, 예측
score = cross_val_score(model, x_train, y_train, cv=kfold)
y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
acc = accuracy_score(y_test, y_predict)

print('acc: ', score, '\n cross_val_score: ', round(np.mean(score),4))
print(y_predict)
print('cross_val_predict ACC :', acc)

# kfold
# cross_val_score : 0.925 
# cross_val_predict ACC : 1.0

# StratifiedKFold
# cross_val_score : 0.9583
# cross_val_predict ACC : 0.9666666666666667