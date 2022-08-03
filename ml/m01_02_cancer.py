from sklearn.model_selection import train_test_split                                     
from sklearn.datasets import load_breast_cancer  
import numpy as np 
from sklearn.svm import LinearSVC


#1. 데이터 
datasets = load_breast_cancer()
x = datasets.data              # x = datasets['data'] 으로도 쓸 수 있다.
y = datasets.target 
print(x.shape, y.shape)        # (569, 30) (569,)
print(np.unique(y, return_counts=True))

x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.8,
    random_state=66
    )

#2. 모델구성
model = LinearSVC()



#3. 컴파일, 훈련
model.fit(x_train, y_train)



#4. 평가, 예측
results = model.score(x_test, y_test)
y_predict = model.predict(x_test)

print('결과 acc :', results)

# loss :  [0.2551944851875305, 0.9122806787490845]   


#머신러닝 사용
# 결과 acc : 0.8947368421052632






