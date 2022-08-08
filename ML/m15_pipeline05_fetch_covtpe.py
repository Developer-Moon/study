from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_covtype
#----------------------------------------------------------------------------------------------------------------#
from sklearn.pipeline import make_pipeline
#----------------------------------------------------------------------------------------------------------------#


#1. 데이터
datasets = fetch_covtype()
x = datasets.data 
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1234)


#2. 모델구성
# model = make_pipeline(MinMaxScaler(), SVC()) 
model = make_pipeline(MinMaxScaler(), RandomForestClassifier()) 


#3. 훈련
model.fit(x_train, y_train) # 스케일이 제공된 상태, fit과 fit_transform이 같이 돌아간다 - fit_transform을 한 다음 fit을 한다 


#4. 평가, 훈련
result = model.score(x_test, y_test)
print('model.score :', result)

# model.score : 0.9861111111111112