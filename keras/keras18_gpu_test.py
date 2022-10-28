import numpy as np                          
import tensorflow as tf

print(tf.__version__)

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if(gpus) :
    print('쥐피유 돌아유')
else:
    print('내가 돌아유')
    
# 밑에 소스 넣고 속도 확인ㄱㄱ