#자연어 처리는 기본적으로 시계열 데이터를 깔고 들어간다(LSTM)

from keras.preprocessing.text import Tokenizer
import numpy as np

text1 = "나는 진짜 매우 매우 맛있는 밥을 엄청 마구 마구 마구 먹었다."
text2 = "나는 지구용사 이재근이다. 멋있다. 또 또 얘기해봐"

token = Tokenizer()
token.fit_on_texts([text1, text2]) # 수치화 작업

print(token.word_index) # {'마구': 1, '매우': 2, '나는': 3, '진짜': 4, '맛있는': 5, '밥을': 6, '엄청': 7, '먹었다': 8}
                        # 반복 횟수가 많은 어절부터 배열

x = token.texts_to_sequences([text1, text2]) # 인덱스화
print(x) # [[2, 5, 3, 3, 6, 7, 8, 1, 1, 1, 9], [2, 10, 11, 12, 4, 4, 13]]




from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
x_new = x[0] + x[1]
print(x_new) # [2, 5, 3, 3, 6, 7, 8, 1, 1, 1, 9, 2, 10, 11, 12, 4, 4, 13] # one hot이 안되서 붙힌다

ohe = OneHotEncoder(categories='auto', sparse= False)

x_new = np.array(sparse=False)
print(x_new)
print(x_new.shape) # (18, 14)



# ohe = OneHotEncoder       one hot으로 수정해라
# x = ohe.fit(x.reshape)
# print(x)