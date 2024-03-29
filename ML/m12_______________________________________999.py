from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
#----------------------------------------------------------------------------------------------------------------#
from sklearn.pipeline import make_pipeline
#----------------------------------------------------------------------------------------------------------------#


#1. 데이터
datasets = load_iris()
x = datasets.data 
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1234)

# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)       
# model에서 Scaler를 사용해서 여기서 할 필요가없다


#2. 모델구성
# model = make_pipeline(MinMaxScaler(), SVC()) 
model = make_pipeline(MinMaxScaler(), RandomForestClassifier()) 


#3. 훈련
model.fit(x_train, y_train) # 스케일이 제공된 상태, fit과 fit_transform이 같이 돌아간다 - fit_transform을 한 다음 fit을 한다 


#4. 평가, 훈련
result = model.score(x_test, y_test)
print('model.score :', result)

# Fitting 5 folds for each of 15 candidates, totalling 75 fits
# 최적의 매개변수 : RandomForestRegressor(max_depth=10, n_estimators=200, n_jobs=-1)
# 최적의 파라미터 : {'n_jobs': -1, 'n_estimators': 200, 'max_depth': 10}
# best_score : 0.8444775518152025
# model.score : 0.8822944263287994
# r2 score : 0.8822944263287993
# best tuned r2 : 0.8822944263287993
# 걸린시간 : 9.58 초