from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC # Support Vector Classifier - 레거시안 사이킷런 모델, 원핫 X, 컴파일 X, argmax X
#1

#1. 데이터
datasets = load_iris()
x = datasets['data']
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=9)


#2. 모델구성
model = LinearSVC() 


#3. 컴파일, 훈련
model.fit(x_train, y_train)


#4. 평가, 예측
results = model.score(x_test, y_test) # evaluate대신 score사용
print('acc :', results)               # 분류모델에서는 accuracy // 회귀모델에서는 R2가 자동

# acc : 1.0
# ML - acc : 1.0 
# 아주 빠르다, 단층 레이어