from sklearn.model_selection import train_test_split                                     
from sklearn.datasets import load_breast_cancer  
from sklearn.svm import LinearSVC


#1. 데이터 
datasets = load_breast_cancer()
x = datasets.data           
y = datasets.target 

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66)


#2. 모델구성
model = LinearSVC()


#3. 컴파일, 훈련
model.fit(x_train, y_train)


#4. 평가, 예측
results = model.score(x_test, y_test)
print('acc :', results)

# acc : 0.9122806787490845
# ML - acc : 0.8947368421052632