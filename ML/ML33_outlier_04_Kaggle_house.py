import numpy as np
aaa = np.array([[-10, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 50],
               [100, 200, -30, 400, 500, 600, -70000, 800, 900, 1000, 210, 420, 350]])

print(aaa.shape) # (2, 13)
aaa = np.transpose(aaa)
print(aaa.shape) # (13, 2)
print(aaa)

from sklearn.covariance import EllipticEnvelope
outliers = EllipticEnvelope(contamination=.2)

outliers.fit(aaa) 
results_02 = outliers.predict(aaa[:,0])
results_01 = outliers.predict(aaa[:,1])

print(results_01)
print(results_02)

# outliers_loc1 = results[:,0]
# outliers_loc2 = outliers(aaa[:,1])
# print('이상치의 위치 :', outliers_loc1)
# print('이상치의 위치 :', outliers_loc2)

# import matplotlib.pyplot as plt
# plt.boxplot(aaa)
# plt.show()