from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd


USE_CUDA = torch.cuda.is_available
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu') # ['cuda:0', 'cuda:1'] 2개 이상일때는 list
# print('torch :', torch.__version__, ' 사용DEVICE :', DEVICE) # torch : 1.12.1  사용DEVICE : cuda:0


#1. 데이터
path = './_data/kaggle_bike/'
train_set = pd.read_csv(path + 'train.csv')
test_set = pd.read_csv(path + 'test.csv')

train_set["hour"] = [t.hour for t in pd.DatetimeIndex(train_set.datetime)]
train_set["day"] = [t.dayofweek for t in pd.DatetimeIndex(train_set.datetime)]
train_set["month"] = [t.month for t in pd.DatetimeIndex(train_set.datetime)]
train_set['year'] = [t.year for t in pd.DatetimeIndex(train_set.datetime)]
train_set['year'] = train_set['year'].map({2011:0, 2012:1})

test_set["hour"] = [t.hour for t in pd.DatetimeIndex(test_set.datetime)]
test_set["day"] = [t.dayofweek for t in pd.DatetimeIndex(test_set.datetime)]
test_set["month"] = [t.month for t in pd.DatetimeIndex(test_set.datetime)]
test_set['year'] = [t.year for t in pd.DatetimeIndex(test_set.datetime)]
test_set['year'] = test_set['year'].map({2011:0, 2012:1})

train_set.drop('datetime',axis=1,inplace=True)
test_set.drop('datetime',axis=1,inplace=True)
train_set.drop('casual',axis=1,inplace=True) 
train_set.drop('registered',axis=1,inplace=True) 

x = train_set.drop(['count'], axis=1)
y = train_set['count']
# x = torch.FloatTensor(x)
# y = torch.LongTensor(y) # 원핫이 필요없다 LongTensorfh 바꿔준다



x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, shuffle=True, random_state=1234)

print(x_train.shape) # (105, 4)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) 



# x_train = torch.FloatTensor(x_train).to(DEVICE)
# x_test = torch.FloatTensor(x_test).to(DEVICE)
# y_train = torch.FloatTensor(y_train).unsqueeze(-1).to(DEVICE)
# y_test = torch.FloatTensor(y_test).unsqueeze(-1).to(DEVICE)

# x_train = torch.FloatTensor(x_train).to(DEVICE)
# x_test = torch.FloatTensor(x_test).to(DEVICE)
# y_train = torch.FloatTensor(y_train).to(DEVICE)
# y_test = torch.FloatTensor(y_test).to(DEVICE)

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_train = torch.FloatTensor(y_train.values).unsqueeze(-1).to(DEVICE)
y_test = torch.FloatTensor(y_test.values).unsqueeze(-1).to(DEVICE)


# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test) # TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.

# x_train = torch.FloatTensor(x_train).to(DEVICE)
# x_test = torch.FloatTensor(x_test).to(DEVICE)

# print(x_train.size())
print(x_train.shape)
print(x_test.shape) # torch.Size([171, 30, 1])
print(y_train.shape)
print(y_test.shape)
print(x_train.dtype)
print(x_test.dtype) # torch.Size([171, 30, 1])
print(y_train.dtype)
print(y_test.dtype)

#2. 모델
model = nn.Sequential(
    nn.Linear(12, 300),
    nn.ReLU(),
    nn.Linear(300, 300),
    nn.ReLU(),
    nn.Linear(300, 128),
    nn.ReLU(),
    nn.Linear(128, 1), # y의 유니크 값이 3개라서
     # softmax를 안해도 된다
).to(DEVICE)

criterion = nn.MSELoss() # criterion = Loss, 

optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, x_train, y_train) :
    # model.train() 안써도 괜찮다
    optimizer.zero_grad() # 처음에 반영되는 가중치를 제거하기 위해 ??
    hypothesis = model(x_train)
    loss = criterion(hypothesis, y_train)
    
    loss.backward()  # 역전파를 실행한다
    optimizer.step() # 가중치를 갱신한다 음수 기울기일때 : w = w - lr * w.grad, 양수 기울기일때 : w = w + lr * w.grad
    
    return loss.item() # train() 에서의 1epochs

EPOCHS = 500
for epochs in range(1, EPOCHS + 1) : 
    loss = train(model, criterion, optimizer, x_train, y_train)
    print('epochs : {}, loss : {:.8f}'.format(epochs, loss)) # 정규표현식 '' {} 메모리 값을 불러오는 공간을 만든 것 그걸 불러 오는 곳이 .foramt(epochs, loss) format안의 순서대로 
                                                             # 
print("======================= 평가, 예측 ======================= ")    
def evaluate(model, criterion, x_test, y_test) :
    model.eval()
    
    with torch.no_grad() :
        hypothesis = model(x_test)
        loss = criterion(hypothesis, y_test)
        return loss.item()
    
loss = evaluate(model, criterion, x_test, y_test)
print('loss :', loss)

y_predict = model(x_test)
# print(y_predict[:10])

# y_predict = y_predict.cpu().numpy()
# y_predict = y_predict.indices
print(y_predict.float())
print(y_test.float())


# score = (y_predict == y_test).float().mean() # 0, 1 개수 가지고 평균을 낸것이 accuracy
# print('r2_score : {:.4f}'.format(score))

from sklearn.metrics import accuracy_score, r2_score
# score = accuracy_score(y_test, y_predict) # gpu상태니까 cpu로 바꿔야 한다
score = r2_score(y_test.cpu().detach().numpy(), y_predict.cpu().detach().numpy())
# score = accuracy_score(y_test.cpu().numpy(), y_predict.cpu().numpy())
print('r2_score :', score) 
# r2_score : 0.9459793912924672
