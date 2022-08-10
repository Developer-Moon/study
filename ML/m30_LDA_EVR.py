from cProfile import label
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, fetch_covtype, load_digits
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
datasets = load_iris()          # (150, 4)  -> (150, 2)
datasets = load_breast_cancer() # (569, 30) -> (569, 1)
datasets = load_wine()          # (178, 13) -> (178, 2)
datasets = fetch_covtype()      # (581012, 54) -> (581012, 6)
datasets = load_digits()        # (1797, 64) -> (1797, 9)

x = datasets.data
y = datasets.target
print(x.shape) # (581012, 54)
#print(np.unique(y, return_counts=True)) # (array([1, 2, 3, 4, 5, 6, 7])

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

lda = LinearDiscriminantAnalysis() # LDA에서 n_components는 1부터 labal값 -1까지 가능 
lda.fit(x, y)
x = lda.transform(x)
print(x.shape)
       

lda_EVR = lda.explained_variance_ratio_ # 변환한 값의 중요도
print(lda_EVR)

cumsum = np.cumsum(lda_EVR) # cumsum - 누적합 하나씩 더해가는걸 보여준다
print(cumsum)