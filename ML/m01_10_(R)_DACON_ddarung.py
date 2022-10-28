from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVR # Support Vector Classifier[회귀] - 레거시안 사이킷런 모델, 원핫 X, 컴파일 X, argmax X
import pandas as pd


#1. 데이터
path = './_data/ddarung/'
train_set = pd.read_csv(path + 'train.csv',index_col=0)
test_set = pd.read_csv(path + 'test.csv', index_col=0)

train_set = train_set.fillna(0)

x = train_set.drop(['count'], axis=1)
y = train_set['count']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)  


#2. 모델구성
model = LinearSVR() # 아주 빠르다, 단층 레이어


#3. 컴파일, 훈련
model.fit(x_train, y_train)


#4. 평가, 예측
result = model.score(x_test, y_test) # evaluate대신 score사용
print('r2 :', result)                # 분류모델에서는 accuracy // 회귀모델에서는 R2가 자동

# ML - r2 : 0.5575541708426989
