import pandas as pd 
import numpy as np
import os



#1 데이터
x = pd.read_csv('C:/study/_data/dacon_plant/train_input/CASE_01.csv')
y = pd.read_csv('C:/study/_data/dacon_plant/train_target/CASE_01.csv')

print(x.shape, y.shape) # (41760, 38) (29, 2)


x_list = np.array(range(29))
all_x_list = []
for i in x_list:
    
    
x_01 = x[:1440]     # (1440, 38)
x_02 = x[1440:2880] # (1440, 38)
x_03 = x[2880:4320] # (1440, 38)


print(x[2880:4320].shape) # 



