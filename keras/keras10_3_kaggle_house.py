# 부동산
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)

from scipy import stats
from scipy.stats import norm, skew #for some statistics


import datetime as dt
from tensorflow.python.keras.models import Sequential   
from tensorflow.python.keras.layers import Dense 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error        

pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points

train = pd.read_csv('./_data/kaggle_house/train.csv')
test = pd.read_csv('./_data/kaggle_house/test.csv')
submit = pd.read_csv('./_data/kaggle_house/sample_submission.csv')
# print(train.shape, test.shape, submit.shape) # (1460, 81) (1459, 80) (1459, 2)
# print(train.head(), test.head())




train = train.drop(['Id'], axis=1) # train셋 index 제거
test = test.drop(['Id'], axis=1)   # test셋 index 제거
# print(train.shape, test.shape)  (1460, 80) (1459, 79)
# print(train.head()) index 빠진거 확인




# 이상치 제거하기(아래 33번째부터 38줄 먼저 그래포로 확인 후 31번째 이상치 제거)
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

label = train['SalePrice']
all_data = pd.concat((train,test)).reset_index(drop=True)
all_data = all_data.drop(['SalePrice'],axis=1)

all_data['PoolQC'] = all_data['PoolQC'].fillna('None')
all_data['MiscFeature'] = all_data['MiscFeature'].fillna('None')

all_data['LotFrontage'].value_counts()
all_data['LotFrontage'] = all_data['LotFrontage'].fillna(all_data['LotFrontage'].median())

for col in ['MSZoning', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType']:
    all_data[col] = all_data[col].fillna(all_data[col].mode()[0])
    
cat_col = all_data.dtypes[all_data.dtypes == 'object'].index
for col in cat_col:
    all_data[col] = all_data[col].fillna('None')
    
all_data.dtypes[(all_data.dtypes == 'int64') | (all_data.dtypes == 'float64')].index

num_col = list(all_data.dtypes[(all_data.dtypes == 'int64') | (all_data.dtypes == 'float64')].index) # 숫자로 된 열이름 추출
for col in num_col:
    all_data[col] = all_data[col].fillna(0)

(all_data.isnull().sum() / len(all_data) * 100).sort_values(ascending=False)[:5]

cat_col[0]

from sklearn.preprocessing import LabelEncoder
lbl = LabelEncoder()
cat_col = list(all_data.dtypes[all_data.dtypes == 'object'].index) # 문자열로 된 열이름 추출
for col in cat_col:
    all_data[col] = lbl.fit_transform(all_data[col].values)
    
    
all_data['MSSubClass'] = all_data['MSSubClass'].astype('category')

# 숫자이지만 건축물의 종류를 구분하는 번호이므로 문자로 변경
all_data['MSSubClass'] = all_data['MSSubClass'].astype('category')
# 월은 대소비교보단 계절성을 나타내기 위해 숫자를 문자로 변경 (범주화)
all_data['MoSold'] = all_data['MoSold'].astype('category')
# all_data['MoSold'].value_counts()

all_data.dtypes


test = test.fillna(0)  # 결측지처리 nan 값에 0 기입   추가코드
# print(all_data)

# test = test.fillna(test.mean())  # 결측지처리 nan 값에 0 기입   추가코드
# train = train.dropna()  




# 학습을 위해 all_data를 train과 test로 다시 분할
train_set = all_data[:len(train)]
test_set = all_data[len(train):]
print(train_set.shape, test_set.shape)

x = train
y = label # train['SalePrice']

x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.8,  # 12 = 124  15까지 했음
    random_state=0
    )

# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


#2. 모델구성
model = Sequential()
model.add(Dense(100,input_dim=79))      
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')               
model.fit(x, y, epochs=1000, batch_size=50)  

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)     

y_predict = model.predict(x_test) 

def RMSE(y_test, y_predict):  
    return np.sqrt(mean_squared_error(y_test, y_predict))  




rmse = RMSE(y_test, y_predict)  
print("RMSE :", rmse)           


