import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

y_true = np.array([100, 2])
y_pred = np.array([200, 102])
# mae : 100.0
# mape : 25.5


# y_true = np.array([100, 200]) 이게 더 좋은 데이터
# y_pred = np.array([200, 300])
# mae : 100.0
# mape : 0.75


mae = mean_absolute_error(y_true, y_pred)
print('mae :', mae)


mape = mean_absolute_percentage_error(y_true, y_pred)
print('mape :', mape) 
