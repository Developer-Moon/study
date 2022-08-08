from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import load_wine
import numpy as np
#----------------------------------------------------------------------------------------------------------------#
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
#----------------------------------------------------------------------------------------------------------------#


#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target

# print(x.shape)
# x = np.delete(x, [7, 8, 9], axis=1)
print(x[1])
# [1.32e+01 1.78e+00 2.14e+00 1.12e+01 1.00e+02 2.65e+00 2.76e+00 2.60e-01 1.28e+00 4.38e+00 1.05e+00 3.40e+00 1.05e+03]

# print(x.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=123)


#2. 모델구성
# model = DecisionTreeClassifier()
model = RandomForestClassifier()
# model = GradientBoostingClassifier()
# model = XGBClassifier()


#3. 훈련
model.fit(x_train, y_train)


#4. 평가, 예측
result = model.score(x_test, y_test)
print('model.score :', result)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('accuracy_score :', acc)
print(model,':',model.feature_importances_)

# DecisionTreeClassifier() : 
# [0.01585625 0.07933324 0.         0.         0.03888017 0.
#  0.12962152 0.         0.         0.         0.         0.27920685
#  0.45710197]
# acc : 0.9166666666666666
# delete column acc :  0.9722222222222222-----------------------w증가


# RandomForestClassifier() :
# [0.03853986 0.0149603  0.07259716 0.07074185 0.00496022 0.01139304    
#  0.03669455 0.12361098 0.00421697 0.00557914 0.01198291 0.00326788  
#  0.012907   0.02905231 0.00498784 0.00303802 0.0082828  0.00654172   
#  0.00464485 0.00482165 0.11273581 0.01951125 0.10605224 0.06330677
#  0.01421158 0.01327938 0.04220101 0.13454106 0.01179073 0.00954914]
# acc : 0.9912280701754386
# delete column acc : 0.9912280701754386-----------------------비슷


# GradientBoostingClassifier() :
# [2.15361206e-05 3.05694258e-02 2.62374196e-04 7.12335438e-04   
#  1.42582399e-04 7.89117524e-04 1.52314084e-02 2.98304338e-01
#  6.28633716e-07 3.83627143e-04 4.73470475e-03 2.44189223e-03
#  4.25123993e-03 6.91911970e-03 1.19750768e-03 6.96122984e-04
#  1.95741533e-03 1.18310803e-05 1.91241026e-03 9.96864999e-04
#  4.02299209e-01 5.34817538e-02 1.92935147e-02 3.72335514e-02
#  2.80063678e-03 1.34175202e-05 2.78814418e-02 8.23353761e-02
#  1.34877651e-03 1.77583947e-03]
# acc : 0.9736842105263158
# delete column acc : 0.9824561403508771-----------------------증가


# XGBClassifier() :
# [0.0000000e+00 1.2473219e-02 0.0000000e+00 1.6165057e-02 4.2501171e-03 
#  4.4334880e-03 3.0821435e-02 1.5752706e-01 3.0083963e-04 2.6863426e-04  
#  2.4044227e-03 4.5414967e-03 1.4079969e-03 5.7522329e-03 7.2954143e-03
#  2.5730273e-03 1.4236500e-03 9.2657527e-04 2.6448383e-03 1.5307434e-03 
#  5.3904897e-01 1.9263266e-02 5.9822440e-02 2.0561621e-02 3.6444392e-03
#  6.2815067e-03 3.1552371e-02 5.9223343e-02 8.9962559e-04 2.9621555e-03]  
# acc : 0.9649122807017544
# delete column acc : 0.9736842105263158-----------------------증가