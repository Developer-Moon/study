#자연어 처리는 기본적으로 시계열 데이터를 깔고 들어간다(LSTM)

from keras.preprocessing.text import Tokenizer
import numpy as np


text = "나는 진짜 매우 매우 맛있는 밥을 엄청 마구 마구 마구 먹었다."

token = Tokenizer()
token.fit_on_texts([text]) # 수치화 작업

print(token.word_index) # {'마구': 1, '매우': 2, '나는': 3, '진짜': 4, '맛있는': 5, '밥을': 6, '엄청': 7, '먹었다': 8}
                        # 반복 횟수가 많은 어절부터 배열

x = token.texts_to_sequences([text]) # 인덱스화
print(x) # [[3, 4, 2, 2, 5, 6, 7, 1, 1, 1, 8]] - "나는 진짜 매우 매우 맛있는 밥을 엄청 마구 마구 마구 먹었다."




from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.preprocessing import OneHotEncoder



#원핫을 한다 3(진짜) + 3(진짜) 은 6(엄청)이 안되게 하려고
ohe = OneHotEncoder(categories='auto',sparse= False)
ohe.fit(x)
x = ohe.transform(x)

print(x)       # [[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]]
print(x.shape) # (1, 11)