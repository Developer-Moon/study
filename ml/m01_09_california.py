from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing 
from sklearn.svm import LinearSVR


#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target                                  

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)


#2. 모델구성
model = LinearSVR()


#3. 컴파일, 훈련
model.fit(x_train, y_train)


#4. 평가 예측
results = model.score(x_test, y_test)
print('r2 :', results)

# r2 : 0.5317547523397925

# 머신러닝 사용 r2 : -3.0337605552808364