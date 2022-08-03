from sklearn.model_selection import train_test_split        
from sklearn.datasets import load_wine, load_digits                             
from sklearn.metrics import r2_score
import numpy as np 
from sklearn.svm import LinearSVC

import tensorflow as tf            
tf.random.set_seed(66)         #텐서플로의 난수표 66번 사용 이 난수는 w = 웨이트


#1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets.target 
print(x.shape, y.shape) # (1797, 64) (1797,)    8x8(64)이미지가 1797개 있다는 말  input_dim = 64
print(np.unique(y, return_counts=True))      # (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180], dtype=int64))    이걸 1797,10으로 변환



x_train, x_test, y_train, y_test = train_test_split(x, y,
    test_size=0.33,
    random_state=72
    )


#2. 모델구성
model = LinearSVC()


#3. 컴파일, 훈련
model.fit(x_train, y_train)



#4. 평가, 예측
results = model.score(x_test, y_test)
y_predict = model.predict(x_test)

print('acc :', results)


# acc스코어 : 0.7171717171717171

#머신러닝 사용 acc : 0.9410774410774411