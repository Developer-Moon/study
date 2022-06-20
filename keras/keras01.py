#1. 데이터
import numpy as np                                 # unmpy = 수치데이터 사용시 사용(배열)
x = np.array([1,2,3])                              
y = np.array([1,2,3])



#2. 모델구성
from tensorflow.keras.models import Sequential     # tensorflow안 keras안 model이라는 폴더에 sequential 임폴트
from tensorflow.keras.layers import Dense          # Dense : 밀도

model = Sequential()                               # 모델은 시퀀셜 모델이다                    
model.add(Dense(4, input_dim=1))                   # 레이어를 추가(밀집층을 추가)  input_dim = input 레이어에 들어가는 데이터의 형태 (input : 1, output : 4)
model.add(Dense(5))                                # model.add(Dense(5)) : node  Sequential이 순차적 이라서 2번째 부턴 input_dim을 안적는다(윗줄 4:input 5:output)
model.add(Dense(3))
model.add(Dense(6))
model.add(Dense(2))
model.add(Dense(1))



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')        #1. 컴파일(compile = 컴퓨터가 알아듣게) mse(오차를 줄이는 식) = 평균제곱오차 > 제곱한걸 나누기 2하겠다  optimizer='adam 최적화는 아담(adam)을 쓴다  
model.fit(x, y, epochs=1900)                       #2. 훈련 = fit   데이터 x와 y를 쓸꺼고 epochs=100 백번을 훈련 시키겠다  loss: 1.1842e-15  이런식이면 e는 맨앞 숫자 1앞에 0이 15개 있다는 말 0.00000.....11842라는 말
# 위  model에는 최종의 갱신된 가중치값이 있다

#4. 평가, 예측
loss = model.evaluate(x, y)   # loss = x와 y를 넣은값을 평가해라            x와 y를 넣은값을 평가(evaluate)한다    이 값을 loss 에 반환한다 
print('loss : ', loss)        # 출력해라  loss : 위의 값을 


result = model.predict([4])   #결과  w 곱하기 4를 한 값이 result
print('4의 예측값은 : ', result)

# ctrl + c = 훈련중 훈련종료
# loss :  0.0
# 4의 예측값은 :  [[4.]]


# 예측값이 안나올때 조절방법 : 훈련량 조절,  레이어 개수 조절, 노드 개수 조절, 측정하는 loss=mse조절 등등
# $$하는 업무  취미 : 하이퍼 파라미터 튜닝작업(#3번도 다 파라미터)    데이터 정제가 제일 중요    특기 : 데이터 전처리가 제일 중요












#숙제 스칼라, 벡터, 디멘션을 정의해야한다







