from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
#----------------------------------------------------------------------------------------------------------------#
import matplotlib.pyplot as plt # simple linear
import seaborn as sns
#----------------------------------------------------------------------------------------------------------------#


#1. 데이터
datasets = load_iris()
print(datasets.feature_names) # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
x = datasets['data']
y = datasets['target']

df = pd.DataFrame(x, columns=[['sepal length ', 'sepal width', 'petal length', 'petal width']])

# print(df)
df['Target(Y)'] = y # y컬럼 추가
print(df) # [150 rows x 5 columns]

print("_____________________________ 상관계수 히트 맵 _____________________________")
print(df.corr())
'''
              sepal length  sepal width petal length petal width  Target(Y
sepal length       1.000000   -0.117570     0.871754    0.817941  0.782561
sepal width       -0.117570    1.000000    -0.428440   -0.366126 -0.426658
petal length       0.871754   -0.428440     1.000000    0.962865  0.949035
petal width        0.817941   -0.366126     0.962865    1.000000  0.956547
Target(Y           0.782561   -0.426658     0.949035    0.956547  1.000000
'''

sns.set(font_scale=1.2)
sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)
# square=True : 셀 반응형 비율 고정, False 비율 비고정
# annot=True  : 각 셀의 상관계수 수치 표기
# cbar=True   : map의 우측에 컬러 가이드 표시

plt.show()