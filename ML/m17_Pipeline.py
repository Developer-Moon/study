from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
#----------------------------------------------------------------------------------------------------------------#
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.decomposition import PCA # 컬럼을 압축한다
#----------------------------------------------------------------------------------------------------------------#


#1. 데이터
datasets = load_iris()
x = datasets.data 
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1234)


#2. 모델구성
# model = make_pipeline(MinMaxScaler(), PCA(), RandomForestClassifier()) # PCA() - 컬럼압축(속도가 빨라진다) 성능에 문제가 없다면 사용
model = Pipeline([('minmax', MinMaxScaler()), ('RF', RandomForestClassifier())])
#                   minmax : 변수명 (다 소문자로)          

#3. 훈련
model.fit(x_train, y_train) # 스케일이 제공된 상태, fit과 fit_transform이 같이 돌아간다 - fit_transform을 한 다음 fit을 한다 


#4. 평가, 훈련
result = model.score(x_test, y_test)
print('model.score :', result)

from sklearn.metrics import accuracy_score 
y_predict = model.predict(x_test) # predict는 make_pipeline을 이용하여 scaler가 적용된 상태다
acc = accuracy_score(y_test, y_predict)
print('accuracy :', acc)

# normal Scaler - acc : 0.9666666666666667
# make_pipeline - model.score : 1.0