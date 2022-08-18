from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans # 비지도학습 y값이 필요없다 
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# 쓰는 이유 라벨이 없을때 
# 특성(컬럼)들을 넣어서 남자냐 여자냐 할 때 n_clusters=2를 넣어서 y라벨을 만든다

datasets = load_breast_cancer()
df = pd.DataFrame(datasets.data, columns=[datasets.feature_names])

datasets.target = np.array(datasets.target)
print(np.unique(datasets.target, return_counts=True))
print(datasets.data.shape)

kmeans = KMeans(n_clusters=2, random_state=100000) # n_clusters=3 라벨의 개수
kmeans.fit(df) # (569, 30)

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

print(accuracy_score(x1, x2)) # 0.14586994727592267



