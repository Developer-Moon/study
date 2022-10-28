from sklearn.covariance import EllipticEnvelope
import numpy as np
aaa = np.array([[-10, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 50],
               [100, 200, -30, 400, 500, 600, -70000, 800, 900, 1000, 210, 420, 350]])

aaa = np.transpose(aaa)

aaa_01 = aaa[:,0]
aaa_02 = aaa[:,1]

print(aaa_01)
aaa_01 = aaa_01.reshape(-1,1)
aaa_02 = aaa_02.reshape(-1,1)
print(aaa_01)

outliers = EllipticEnvelope(contamination=.2)

outliers.fit(aaa_01) # 2차원으로 들어가야해서 reshape해줘야한다 line=11
outliers.fit(aaa_02) 
results_01 = outliers.predict(aaa_01)
results_02 = outliers.predict(aaa_02)

print(results_01) # [1 1 1 1 1 1 1 1 1 1 1 1 1]
print(results_02) # [ 1  1  1  1  1  1 -1  1 -1 -1  1  1  1]

