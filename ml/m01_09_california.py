from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score 
from sklearn.datasets import fetch_california_housing 

from sklearn.svm import LinearSVR


#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target                                  

# print(x.shape, y.shape)        # (20640, 8) (20640,)    
# print(datasets.feature_names)  # ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
# print(datasets.DESCR)          # Number of Instances: 20640(행) 

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)



#2. 모델구성
model = LinearSVR()


#3. 컴파일, 훈련
model.fit(x_train, y_train)


#4. 평가 예측
results = model.score(x_test, y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

print('r2 :', results)

# loss : 0.6425122618675232
# r2스코어 : 0.5317547523397925

# 머신러닝 사용 r2 : -3.7280966592065496