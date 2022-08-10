from sklearn.experimental import enable_iterative_imputer # IterativeImputer 이 라이브러리도 아직 실험중
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer # KNNImputer : 비지도학습의 클러스터방식의 학습
import pandas as pd
import numpy as np

data = pd.DataFrame([[2, np.nan, 6, 8, 10],
                      [2, 4, np.nan, 8, np.nan],
                      [2, 4, 6, 8, 10],
                      [np.nan, 4, np.nan, 8, np.nan]])

data = data.transpose()
data.columns = ['x1', 'x2', 'x3', 'x4']
print(data) 

# imputer = SimpleImputer() # 평균값
# imputer = SimpleImputer(strategy='mean')
# imputer = SimpleImputer(strategy='median')
# imputer = SimpleImputer(strategy='most_frequent') # 가장 빈번하게 쓰인걸 쓰고 동일할때는 앞에껄 쓴다
# imputer = SimpleImputer(strategy='constant') # 상수를 넣는데 디폴트가 0
# imputer = SimpleImputer(strategy='constant', fill_value=777) # 그 디폴트를 777로
# imputer = KNNImputer() # 디폴트는 
imputer = IterativeImputer() # 

imputer.fit(data)1
data2 = imputer.transform(data)
print(data2)