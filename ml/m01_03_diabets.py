from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.datasets import load_diabetes  

from sklearn.svm import LinearSVR   


#1. 데이터                        
datasets = load_diabetes()                      
x = datasets.data
y = datasets.target 
 
print(x.shape, y.shape)         # (442, 10) (442,)     컬럼 = 10 스칼라 = 422
print(datasets.feature_names)   # ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
print(datasets.DESCR) 

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=72)



#2. 모델구성
model = LinearSVR()



#3. 컴파일, 훈련
model.fit(x_train, y_train)



#4. 평가 예측
results = model.score(x_test, y_test)
y_predict = model.predict(x_test)



print('acc :', results)


# loss : 2189.79541015625
# r2스코어 : 0.634477819866959


# 머신러닝 사용
# acc : -0.4805696667781383