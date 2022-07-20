import pandas as pd
import numpy as np

df = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                   columns=['a', 'b', 'c'])


print(df)
print('--------------------')
print(df.values)
print(df.to_numpy())
print(np.array(df))