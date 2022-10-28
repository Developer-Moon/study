from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
#----------------------------------------------------------------------------------------------------------------#
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression # (논리회귀)이진분류
#----------------------------------------------------------------------------------------------------------------#


#1. 데이터
datasets = load_breast_cancer()
x, y = datasets.data, datasets.target

print(x.shape, y.shape) # (569, 30) (569,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=123, stratify=y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
# Please also refer to the documentation for alternative solver options: 
#     https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
#   n_iter_i = _check_optimize_result(   
# 스케일링을 하지 않으면 에러가 뜬다



#2. 모델구성
model = BaggingClassifier(LogisticRegression(),   # bagging(Bootstrap Aggregating) : 한 가지 모델을 여러번 돌린다
                          n_estimators=100,       # voting : 여러가지 모델을 여러번 돌림
                          n_jobs=-1,              # bagging classifier에 디시전 트리 넣으면 RandomForest가 된다
                          random_state=123,
                          )


#3. 훈련
model.fit(x_train, y_train)


#4. 평가, 예측
print(model.score(x_test, y_test)) # 0.9736842105263158