from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.datasets import load_digits


# 1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test =  train_test_split(x, y, train_size=0.9, shuffle=True, random_state=86)



# 2. 모델구성
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron, LogisticRegression # !논리회귀(분류임)!
from sklearn.neighbors import KNeighborsClassifier # 최근접 이웃
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier # 결정트리를 여러개 랜덤으로 뽑아서 앙상블해서 봄




# 3. 컴파일, 훈련


# 4. 평가, 예측

# model = LinearSVC()
# model.fit(x_train, y_train )
# score = model.score(x_test, y_test)
# print('acc : ', score) # acc :  0.9444444444444444

# acc score:  0.9611111111111111
# acc score:  0.9388888888888889


model_list = [LinearSVC, Perceptron, LogisticRegression, KNeighborsClassifier, DecisionTreeClassifier, RandomForestClassifier]

for x in model_list:
     model = x()
     model.fit(x_train, y_train)
     score = model.score(x_test, y_test)
     model_name = str(model)
     print(model_name, 'acc: ', score) 
     




"""
model = LinearSVC()
model.fit(x_train, y_train )
score = model.score(x_test, y_test)
print('acc : ', score) # acc :  0.9444444444444444

model = Perceptron()
model.fit(x_train, y_train )
score = model.score(x_test, y_test)
print('acc : ', score) # acc :  0.9388888888888889

model = LogisticRegression()
model.fit(x_train, y_train )
score = model.score(x_test, y_test)
print('acc : ', score) # acc :  0.9611111111111111

model = KNeighborsClassifier()
model.fit(x_train, y_train )
score = model.score(x_test, y_test)
print('acc : ', score) # acc :  0.9944444444444445

model = DecisionTreeClassifier()
model.fit(x_train, y_train )
score = model.score(x_test, y_test)
print('acc : ', score) # acc :  0.8611111111111112

model = RandomForestClassifier()
model.fit(x_train, y_train )
score = model.score(x_test, y_test)
print('acc : ', score) # acc :  0.9666666666666667

"""

