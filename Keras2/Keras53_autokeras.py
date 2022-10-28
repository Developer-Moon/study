import tensorflow as tf
import autokeras as ak
print(ak.__version__)
import keras
import time


#1. 데이터
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)


#2. 모델
model = ak.ImageClassifier(
    overwrite=True, # 기본값은 TRUE, FALSE인 경우 동일한 이름의 기존 프로젝트가 있으면 다시 로드합니다. 그렇지 않으면 프로젝트를 덮어쓴다
    max_trials=2    # 1epochs당 2번  
)

#3. 컴파일, 훈련
start = time.time()
model.fit(x_train, y_train, validation_split=0.2, epochs=5)
end = time.time()


#4. 평가, 예측
y_predict = model.predict(x_test)
results = model.evaluate(x_test, y_test)
print('결과: ', results)
print('시간: ', round(end-start, 4))

# 결과 : [0.03845507651567459, 0.9872999787330627] - loss ,acc
# 시간 : 4161.5247