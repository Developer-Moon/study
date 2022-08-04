from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC # Support Vector Classifier - 레거시안 사이킷런 모델 


#1. 데이터
datasets = load_iris()
x = datasets['data']
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=9)


#2. 모델구성
model = LinearSVC() 


#3. 컴파일, 훈련
model.fit(x_train, y_train) # 원핫이 필요없다 훈련은 통상 100번, 컴파일이 포함


#4. 평가, 예측
results = model.score(x_test, y_test) # evaluate대신 score
print('acc :', results)

# acc : 1.0

# 머신러닝 사용 - acc : 1.0