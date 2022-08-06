from sklearn.model_selection import train_test_split 
from sklearn.datasets import load_boston  
from sklearn.svm import LinearSVR # Support Vector Regressor - 레거시안 사이킷런 모델


#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target      
                  
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=66)


#2. 모델구성
model = LinearSVR() 


#3. 컴파일, 훈련
model.fit(x_train, y_train)


#4. 평가, 예측
results = model.score(x_test, y_test)
print('r2 :', results)

# r2 : 0.738325009535564
# ML - r2 : 0.42471395829129654