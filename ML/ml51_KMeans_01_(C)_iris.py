from sklearn.cluster import KMeans # 비지도학습 y값이 필요없다 - 회귀는 X
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

# 쓰는 이유 라벨이 없을때 
# 특성(컬럼)들을 넣어서 남자냐 여자냐 할 때 n_clusters=2를 넣어서 y라벨을 만든다

datasets = load_iris()
df = pd.DataFrame(datasets.data, columns=[datasets.feature_names])

datasets.target = np.array(datasets.target)
print(np.unique(datasets.target, return_counts=True)) # (array([0, 1, 2]), array([50, 50, 50], dtype=int64))

kmeans = KMeans(n_clusters=3, random_state=1234) # n_clusters=3 라벨의 개수
kmeans.fit(df)

print(kmeans.labels_) # ->  생성된 y값
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 1 2 2 2 2 1 2 2 2 2
#  2 2 1 1 2 2 2 2 1 2 1 2 1 2 2 1 1 2 2 2 2 2 1 2 2 2 2 1 2 2 2 1 2 2 2 1 2
#  2 1]

print(datasets.target) # ->  원래의 y값
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
#  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#  2 2]


df['clusters'] = kmeans.labels_
df['target'] = datasets.target
x1 = df['clusters']
x2 = df['target']

print(df)

print(accuracy_score(x1, x2)) # 0.8933333333333333
