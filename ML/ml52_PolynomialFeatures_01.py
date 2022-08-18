from sklearn.preprocessing import PolynomialFeatures # 전처리 단계에서 증폭 개념 - [단항을 -> 다항으로]
import pandas as pd
import numpy as np
'''
x = np.arange(8).reshape(4, 2)

print(x)
# [[0 1]
#  [2 3]
#  [4 5]
#  [6 7]]
print(x.shape) # (4, 2)

pf = PolynomialFeatures(degree=3)  # 앞에 1은 무조건 넣고, 자기꺼, 자기꺼,  2의제곱, 2곱하기3, 3의제곱
x_pf = pf.fit_transform(x)

print(x_pf)
# [[ 1.  0.  1.  0.  0.  1.]
#  [ 1.  2.  3.  4.  6.  9.]
#  [ 1.  4.  5. 16. 20. 25.]
#  [ 1.  6.  7. 36. 42. 49.]]

print(x_pf.shape) # (4, 6)

'''


x = np.arange(12).reshape(4, 3)

print(x)
# [[ 0  1  2]
#  [ 3  4  5]
#  [ 6  7  8]
#  [ 9 10 11]]
print(x.shape) # (4, 3)
pf = PolynomialFeatures(degree=2)  #통상 2까지 넣는다
x_pf = pf.fit_transform(x)


print(x_pf)
# [[  1.   0.   1.   2.   0.   0.   0.   1.   2.   4.]
#  [  1.   3.   4.   5.   9.  12.  15.  16.  20.  25.]
#  [  1.   6.   7.   8.  36.  42.  48.  49.  56.  64.]
#  [  1.   9.  10.  11.  81.  90.  99. 100. 110. 121.]]
print(x_pf.shape) # (4, 6)