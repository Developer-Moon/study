from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_covtype
from sklearn.svm import LinearSVC


#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target 

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=72)


#2. 모델구성
model = LinearSVC()


#3. 컴파일, 훈련
model.fit(x_train, y_train)


#4. 평가, 예측
results = model.score(x_test, y_test)
print('acc :', results)

# acc : 0.5815765591913797
# ML - acc : 0.5494069909353584