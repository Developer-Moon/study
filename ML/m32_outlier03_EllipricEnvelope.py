from sklearn.covariance import EllipticEnvelope
import numpy as np

aaa = np.array([[-10, 2, 3, 4, 5, 6, 700, 8, 9, 10, 11, 12, 50]])
aaa = aaa.reshape(-1, 1)

outliers = EllipticEnvelope(contamination=.1) # contamination : 이상치(오염도)
                                              # 데이터의 10프로를 이상치로 잡겠다 [이상치의 위치를 찾아준다]

outliers.fit(aaa) 
results = outliers.predict(aaa)
print(results)