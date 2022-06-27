#1. 데이터 
import numpy as np                                 # unmpy(=Numerical Python, 파이썬 라이브러리) : 벡터 및 행렬 연산에 있어서 매우 편리한 기능을 제공
x = np.array([1,2,3])                              # numpy에서는 기본적으로 array(배열, 행렬)라는 단위로 데이터를 관리하며 이에 대해 연산을 수행
y = np.array([1,2,3])



#2. 모델구성
from tensorflow.keras.models import Sequential     # 모듈을 불러온다 - 1. import 모듈:모듈 전체를 가져옴, 2. from 모듈 import 이름:모듈속 이름의 파일만 가져옴                        
from tensorflow.keras.layers import Dense          # tensorflow안 keras안 layers이라는 폴더에 Dense를 inport    Dense : 밀도

model = Sequential()                               # 시퀀셜 모델(순차적인 모델)                  
model.add(Dense(4, input_dim=1))                   # output:4 input:1 인 처음 layer - 모델의 시작 
model.add(Dense(5))                                # 5개의 노드(뉴런)로 이루어진 layer - 2번째 부턴 input_dim 사용X(윗줄 4:input 5:output)
model.add(Dense(3))                                # 3개의 노드(뉴런)로 이루어진 layer
model.add(Dense(6))
model.add(Dense(2))
model.add(Dense(1))



#3. 컴파일(compile =컴퓨터가 알아듣게), 훈련
model.compile(loss='mse', optimizer='adam')        # 컴파일 mse(오차를 줄이는 식)=평균제곱오차 > 제곱한걸 나누기 2하겠다  optimizer='adam 최적화는 adam을 쓴다  
model.fit(x, y, epochs=100)                        # 훈련(fit) 데이터 x와 y를 쓸꺼고 epochs=10번을 훈련  loss: 1.1842e-15 e는 맨앞 숫자 1앞에 0이 15개 있다(0.00000..118422)
# 위  model에는 최종의 갱신된 가중치값이 있다


#4. 평가, 예측
loss = model.evaluate(x, y)                        # loss = x와 y를 넣은값을 평가(evaluate)해라 [이 값을 loss 에 반환한다]
print('loss : ', loss)                             # 출력해라(print)  loss : 위의 값을 


result = model.predict([4])                        # result=W x 4를 한 값       
print('4의 예측값은 : ', result)

# ctrl + c = 훈련중 훈련종료
# loss :  0.0
# 4의 예측값은 :  [[4.]]


# 예측값이 안나올때 조절방법 : 훈련량 조절,  레이어 개수 조절, 노드 개수 조절, 측정하는 loss=mse조절 등등
# $$하는 업무  취미 : 하이퍼 파라미터 튜닝작업(#3번도 다 파라미터)    데이터 정제가 제일 중요    특기 : 데이터 전처리가 제일 중요



