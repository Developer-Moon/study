from cProfile import label
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, fetch_covtype, load_digits, load_diabetes
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xg
print('xgboost version :', xg.__version__) # 1.6.1

################################################################################
# 비지도학습을 통한 PCA - column만 압축 [회귀, 분류]
# 지도학습을 통한 LDA - column과 labal을 같이 압축 [분류]

# StandardSclear 를 쓰고 PCA를 쓰는 경우에 잘 나오는 경우가 있다고 한다(확인해봐라)
################################################################################


#1. 데이터
path = './_data/kaggle_titanic/'
train_set = pd.read_csv(path + 'train.csv')
test_set = pd.read_csv(path + 'test.csv')

train_set = train_set.drop(columns='Cabin', axis=1)
train_set['Age'].fillna(train_set['Age'].mean(), inplace=True)
train_set['Embarked'].fillna(train_set['Embarked'].mode()[0], inplace=True)
train_set.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)

y = train_set['Survived']
x = train_set.drop(columns = ['PassengerId','Name','Ticket','Survived'], axis=1)
y = np.array(y).reshape(-1, 1)

print(x.shape) # (891, 7)
print(np.unique(y, return_counts=True)) # ([549, 342])

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

lda = LinearDiscriminantAnalysis(n_components=1) # LDA에서 n_components는 1부터 labal값 -1까지 가능
lda.fit(x, y)
x = lda.transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,shuffle=True, random_state=9, stratify=y) # stratify=y y라벨의 비율을 일정하게 잡아준다.)

# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)


#2. 모델구성
from xgboost import XGBClassifier
model = XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor', gpu_id='0')



#3. 컴파일, 훈련
import time
start = time.time()
model.fit(x_train, y_train)
end = time.time()

#4. 평가, 예측
results = model.score(x_test, y_test)
print('결과 :', results)
print('time :', end - start)



# 결과 : 0.8681703570475805
# time : 114.03381443023682

# xgboost - GPU n_components : 10
# 결과 : 0.8406065247885166
# time : 15.830528736114502


# LDA - column:54, array:7, n_components=9 
# 결과 : 0.7868385497792656
# time : 4.130921363830566

# LDA - column:54, array:7, n_components=1 n_components를 줄일수록 점점 낮아진다
# 결과 : 0.7018493498446684
# time : 2.376640558242798