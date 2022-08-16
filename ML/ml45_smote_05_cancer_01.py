from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
# pip insatll imblearn
from imblearn.over_sampling import SMOTE 
import pandas as pd
import numpy as np

# 스모트 넣어서

#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets['target']
print(x.shape, y.shape)                 # (178, 13) (178,)
print(type(x))                          # <class 'numpy.ndarray'>
print(np.unique(y, return_counts=True)) # (array([0, 1]), array([212, 357], dtype=int64))
print(pd.Series(y).value_counts())      # pandas 
