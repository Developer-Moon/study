import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier,XGBRegressor
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, accuracy_score

### 1.데이터 ###
path  = './_data/dacon_travel/' 

train = pd.read_csv( path + 'train.csv', index_col=0)
test = pd.read_csv(path + 'test.csv', index_col=0)
submit = pd.read_csv(path + 'sample_submission.csv',index_col=0)

print(train.shape, test.shape) # (1955, 19) (2933, 18)


##################### 라벨인코더 ######################
le = LabelEncoder()

idxarr = train.columns
idxarr = np.array(idxarr)

for i in idxarr:
      if train[i].dtype == 'object':
        train[i] = le.fit_transform(train[i])
        test[i] = le.fit_transform(test[i])


### 상관관계 ###
sns.set(font_scale= 0.8 )
sns.heatmap(data=train.corr(), square= True, annot=True, cbar=True) # square: 정사각형, annot: 안에 수치들 ,cbar: 옆에 bar

plt.show() 
train.to_csv(path + 'train22.csv',index=False)
