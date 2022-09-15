import numpy as np
from keras.models import Sequential
from keras.layers import Dense


#1. 데이터
import numpy as np
import torch

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

USE_CUDA = torch.cuda.is_available()                   
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')
print(torch.__version__, '사용DVICE:', DEVICE)
# device를 하는건 model이랑 데이터에만 붙힌다


# 1. data
x_train = np.array([1, 2, 3, 4, 5, 6, 7]) # (7, )
x_test = np.array([8, 9, 10])             # (3, )
y_train = np.array([1, 2, 3, 4, 5, 6, 7]) # (7, )
y_test = np.array([8, 9, 10])             # (3, )
x_ptrdict = np.array(11, 12, 13)

x_train = torch.FloatTensor(x_train).to(DEVICE) # cpu에 있는 데이터를 gpu로 보내는데 
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_train = torch.FloatTensor(y_train).to(DEVICE)
y_test = torch.FloatTensor(y_test).to(DEVICE)
x_ptrdict = torch.FloatTensor(x_ptrdict).to(DEVICE)


###### 스케일링 ######
x_test= (x_test - torch.mean(x_train)) / torch.std(x_train) # 
x_train = (x_train - torch.mean(x_train)) / torch.std(x_train)           # 

# print(x.shape, y.shape, x_test.shape)
                                    
# 2. model
model = nn.Sequential(
    nn.Linear(3,4),
    nn.Linear(4,5),
    nn.Linear(5,3),
    # nn.ReLU(),          # 위에 적용됨
    nn.Linear(3,2),
    nn.Linear(2,2)
).to(DEVICE)

# 3. compile, fit
optimizer = optim.SGD(model.parameters(), lr=0.001) # model.parameters() 모델에 있는 파라미터로 넣어달라

def train(model, optimizer, x, y):
    optimizer.zero_grad() # 기울기 값을 초기화 하는 이유
    
    hypothesis = model(x)
    loss = F.mse_loss(hypothesis, y) # model에 x를 넣은걸 hypothesis y값의 차이를 반환
    
    loss.backward() # loss로 역전파를 하겠다
    optimizer.step() # 그 값을 옵티마이져로 가중치를 업데이트 하겠다

    return loss.item()
    
epochs = 12000
for epoch in range(1, epochs+1):
    loss = train(model, optimizer, x_train, y_train)
    print('epoch: {}, loss: {}'.format(epoch, loss))
    
# 4. eval
def evaluate(model, x, y): 
    model.eval() # 평가모드
    
    with torch.no_grad(): # 가중치를 갱신할 필요가 없으니까 
        x_predict = model(x)
        results = nn.MSELoss()(x_predict, y)
    
    return results.item()

result_loss = evaluate(model, x_test, y_test)
print(f'최종 loss: {result_loss}')

results = model(x_ptrdict)
results = results.cpu().detach().numpy()
print(f'예측값: {results}')

# 예측: [[9, 30, 210]] -> 예상 y값 [[10, 1.9]]

# 최종 loss: 4.7168261517072096e-05
# 예측값: [[10.004651   1.9085085]]
