from sklearn.model_selection import train_test_split                                     
from sklearn.datasets import load_wine 
from sklearn.svm import LinearSVC


#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=72)


#2. 모델구성
model = LinearSVC()

                           
#3. 컴파일, 훈련
model.fit(x_train, y_train)


#4. 평가, 예측
results = model.score(x_test, y_test)
print('acc :', results)

# acc : 0.5932203389830508

# 머신러닝 사용 - acc :  0.847457627118644



