from lib2to3.pgen2.tokenize import TokenError
from keras.preprocessing.text import Tokenizer
import numpy as np            
               


#1. 데이타
docs = ['너무 재밌어요', '참 괴오예요', '참 잘 만든 영화에요', '추천하고 싶은 영화힙니다.', '한 번 더 보고 싶네요', '글쎄요',
        '별로에요', '생각보다 지루해요', '연기가 어색해요', '재미없어요', '너무 재미없다', '참 재밋네요', '민수가 못 생기긴 했어요', '안결 혼해요']

# 긍정1, 부정0
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1,0]) # (14,0) y값

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)

"""
{'참': 1, '너무': 2, '재밌어요': 3, '괴오예요': 4, '잘': 5, '만든': 6, '영화에요': 7, '추천하고': 8, '싶은': 9, '영화힙니다': 10,
'한': 11, '번': 12, '더': 13, '보고': 14, '싶네요': 15, '글쎄요': 16, '별로에요': 17, '생각보다': 18, '지루해요': 19, '연기가': 20,
'어색해요': 21, '재미없어요': 22, '재미없다': 23, '재밋네요': 24, '민수가': 25, '못': 26, '생기긴': 27, '했어요': 28, '안결': 29, '혼해요': 30}
"""

x = token.texts_to_sequences(docs)
print(x) # [[2, 3], [1, 4], [1, 5, 6, 7], [8, 9, 10], [11, 12, 13, 14, 15], [16], [17], [18, 19], [20, 21], [22], [2, 23], [1, 24], [25, 26, 27, 28], [29, 30]]
# 길이를 맞추기 위해 수치가 가장 큰 것에 맞춰 작은애들을 0으로 채워준다
# ex) 만약 큰 놈이 30만개라면 30만개 짜리를 버린다


from keras.preprocessing.sequence import pad_sequences # 0으로 채울라고 쓰는 모듈
# LSTM특성상 뒤에 있는 데이터가 중요하므로 통상 0을 앞에서부터 채운다
# 0을 뒤에서부터 채우면 뒤로 갈수록 0에 수렴한다

pad_x = pad_sequences(x, padding='pre', maxlen=5) # 0을 앞에서 부터 채우고 최대 5글자까지만 하겠습니다. 최대가 5라서 [11, 12, 13, 14, 15]
print(pad_x)
print(pad_x.shape) # (14, 5) 보고 1: Dense모델, 2 : reshape하여 3차원변형 후 LSTM

word_size = len(token.word_index)
print('word_size : ', word_size) # 단어사전의 개수 : 30 

print(np.unique(pad_x, return_counts=True)) # padding을 해줄떄 0번째 자리도 해줘서 31개가 된다
# (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30])
# array([37,  3,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1]

#여기서 원핫을 해야 하는데 하지 않고 대신 하단에서 임베딩한다

#0000 별로에요???


#2. 모델구성
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Embedding



model = Sequential() 
                  # 단어사전의 개수 아웃풋 노드의개수   길이 5
model.add(Embedding(input_dim=5, output_dim=11, input_length=5)) # 임베딩을 거치면 통상 3차원이 나온다 
# model.add(Embedding(input_dim=33, output_dim=100, input_length=5)) # 임베딩을 거치면 통상 3차원이 나온다
                                            # input_length를 안다면 명시해주고 아니면 none으로 들어간다
# input_dim이 shape과 다르면 param수는 달라진 상태에서 모델이 돌아가지만 사용불가능

# model.add(Embedding(33, 10))  # 이렇게도 가능 단어사전의개수 33개 아웃풋노드10개
# model.add(Embedding(31, 10, 5))            # error
# model.add(Embedding(31, 10, input_length=5)) # input_length는 써줘야한다 (10개짜리 5묶음) 
# Embedding 단어의 상관관계에 맞춰 백터로 수치화
model.add(LSTM(32))                            
model.add(Dense(1, activation='sigmoid')) # 0, 1을 구분해서 sigmoid
model.summary()





#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(pad_x, labels, epochs=3, batch_size=16)



#4. 평가, 예측
# acc = model.evaluate(pad_x, labels)[0] loss
acc = model.evaluate(pad_x, labels)[1] # acc

print('acc :', acc)
# acc : 0.5
# acc : 0.6922626495361328
#
