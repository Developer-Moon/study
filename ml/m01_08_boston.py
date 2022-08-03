from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score 
from sklearn.datasets import load_boston  
from sklearn.svm import LinearSVR               # 서포트 백터 리그래스    레거시안 사이킷런 모델 

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target      
                  
print(x.shape, y.shape)         # (506, 13) (506,) - 데이터 : 506개,   컬럼 : 13 - input_dim (506개의 스칼라 1개의 벡터)
print(datasets.feature_names)   # ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT'] # 열 별로 이름
print(datasets.DESCR)           # 컬럼들의 소개가 나온다




x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)



#2. 모델구성
# model = Sequential()
# model.add(Dense(30, input_dim=13))
# model.add(Dense(20))
# model.add(Dense(10))
# model.add(Dense(10))
# model.add(Dense(10))
# model.add(Dense(1))
model = LinearSVR() 


#3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')                   
# model.fit(x_train, y_train, epochs=300, batch_size=10)
model.fit(x_train, y_train)




#4. 평가 예측
results = model.score(x_test, y_test)  # evaluate대신 score    # 회귀에서는 r2_score 분류에서는 accuracy - auto 설정이라 이해??
y_predict = model.predict(x_test)





print('predict :', y_predict)
print('결과 r2 :', results)






# loss : 21.61882972717285
# r2 : 0.738325009535564

