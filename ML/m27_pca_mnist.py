from sklearn.decomposition import PCA
from keras.datasets import mnist
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# PCA를 통해 0.95 이상인 n_composition은 몇개?
# 0.95
# 0.99
# 0.999
# 1.0
# 힌트 : np.argmax

(x_train, _), (x_test, _) = mnist.load_data() # _ = 안 들고오겠다

print(x_train.shape, x_test.shape) # (60000, 28, 28) (10000, 28, 28)

x = np.append(x_train, x_test, axis=0)
print(x.shape) # (70000, 28, 28)



x = x.reshape(70000, 28*28) # (70000, 784)
print(x.shape)

pca = PCA(n_components=784)
x = pca.fit_transform(x)

pca_EVR = pca.explained_variance_ratio_

cumsum =np.cumsum(pca_EVR)
print(cumsum)

print(np.argmax(cumsum >= 0.95) + 1)  # 154
print(np.argmax(cumsum >= 0.99) + 1)  # 331
print(np.argmax(cumsum >= 0.999) + 1) # 486
print(np.argmax(cumsum >= 1.0) + 1)   # 713












'''
x = x.reshape(70000, 28*28) # (70000, 784)

pca = PCA(n_components=154)
x = pca.fit_transform(x)
print(x.shape)

pca_EVR = pca.explained_variance_ratio_ # 변환한 값의 중요도
print(pca_EVR)

print(sum(pca_EVR)) # 0.9999999203185791 상위 전체의 합 : 1에 가깝다

cumsum = np.cumsum(pca_EVR) # cumsum - 누적합 하나씩 더해가는걸 보여준다
print(cumsum)

import matplotlib.pyplot as plt
plt.plot(cumsum)
plt.grid()
plt.show()

'''


