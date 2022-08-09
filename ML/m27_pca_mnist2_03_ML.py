from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score 
from sklearn.decomposition import PCA
from keras.datasets import mnist
import tensorflow as tf
import pandas as pd
import numpy as np
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier


(x_train, y_train), (x_test, y_test) = mnist.load_data() # _ = 안 들고오겠다

x = np.append(x_train, x_test, axis=0)
print(x.shape) # (70000, 28, 28)

x = x.reshape(70000, 28*28) # (70000, 784)
print(x.shape)

pca = PCA(n_components=154)
x = pca.fit_transform(x)

pca_EVR = pca.explained_variance_ratio_

cumsum =np.cumsum(pca_EVR)
# print(cumsum)
# print(np.argmax(cumsum >= 0.996) + 1)  

x_train = x[:60000] # (60000, 403)
x_test = x[60000:]  # (10000, 403)


#2. 모델구성
model = GradientBoostingClassifier(verbose=1)


#3. 컴파일, 훈련
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time() - start_time

#4. 결과, 예측
result = model.score(x_test, y_test)
print('model.score :', result)

# cum_list = [154, 331, 486, 713]

# DecisionTreeClassifier - 154
# model.score : 0.8348

# RandomForestClassifier - 154
# model.score : 0.9481

# GradientBoostingClassifier - 154




# valueError : Invalid classes inferred from unique values of 'y'. Expected : [0 1 2 3 4 5 6], got [1 2 3 4 5 6 7 ]
# 해결법 train_test_split에서 strafify=y

# xgboost가 느리다 GPU쓸란다 할때
# model = XGBClassifier(여기에다가)
# tree_method='gpu_hist', predictor='gpu_predctor', gpu_id=0,