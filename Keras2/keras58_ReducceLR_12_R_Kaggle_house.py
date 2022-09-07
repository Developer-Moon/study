from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Input, Dropout, Conv2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.datasets import fetch_california_housing
from tqdm import tqdm_notebook
import pandas as pd
import numpy as np
import time


#1. Data

path = './_data/kaggle_house/'
train_set = pd.read_csv(path + 'train.csv', index_col=0)
test_set = pd.read_csv(path + 'test.csv', index_col=0)

drop_cols = ['Alley', 'PoolQC', 'Fence', 'MiscFeature'] 

test_set.drop(drop_cols, axis = 1, inplace =True)
train_set.drop(drop_cols, axis = 1, inplace =True)

cols = ['MSZoning', 'Street','LandContour','Neighborhood','Condition1','Condition2',
                'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Foundation',
                'Heating','GarageType','SaleType','SaleCondition','ExterQual','ExterCond','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',
                'BsmtFinType2','HeatingQC','CentralAir','Electrical','KitchenQual','Functional',
                'FireplaceQu','GarageFinish','GarageQual','GarageCond','PavedDrive','LotShape',
                'Utilities','LandSlope','BldgType','HouseStyle','LotConfig']

for col in tqdm_notebook(cols):
    le = LabelEncoder()
    train_set[col]=le.fit_transform(train_set[col])
    test_set[col]=le.fit_transform(test_set[col])

train_set = train_set.fillna(train_set.mean()) 
test_set = test_set.fillna(test_set.mean())

x = train_set.drop(['SalePrice'], axis=1)
y = train_set['SalePrice']

print(x.shape, y.shape)                 # (1460, 75) (1460,)    
print(np.unique(y, return_counts=True))

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, shuffle=True, random_state=68)


#2. Model
activation='relu'
drop=0.2
optimizer='adam'

inputs = Input(shape=(12), name='input')
x = Dense(64, activation=activation, name='hidden1')(inputs)
x = Dense(128, activation=activation, name='hidden2')(x)
x = Dense(128, activation=activation, name='hidden3')(x)
x = Dense(128, activation=activation, name='hidden4')(x)
outputs = Dense(1, activation=activation)(x)
model = Model(inputs=inputs, outputs=outputs)


#3. Compile
model.compile(optimizer=optimizer, metrics=['mse'], loss='mse')
es= EarlyStopping(monitor='val_loss', patience=50, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=20, mode='auto', verbose=1, factor=0.5) # factor=0.5 : 50% 만큼 lr을 감소 시킨다  디폴트 lr은 0.001
start = time.time()
model.fit(x_train, y_train, epochs=500, validation_split=0.4, batch_size=128, callbacks=[es, reduce_lr])
end = time.time() - start


#4. Result
loss, r2 = model.evaluate(x_test, y_test)
y_pred = model.predict(x_test)
print('Time :', end)
print('Loss :', loss)
print('R2 :', r2)


