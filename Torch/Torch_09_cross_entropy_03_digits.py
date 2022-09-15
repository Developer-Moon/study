from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import numpy as np


USE_CUDA = torch.cuda.is_available
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu') # ['cuda:0', 'cuda:1'] 2개 이상일때는 list
# print('torch :', torch.__version__, ' 사용DEVICE :', DEVICE) # torch : 1.12.1  사용DEVICE : cuda:0


#1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets['target']

print(np.unique(y, return_counts=True)) # (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180], dtype=int64))
# x = torch.FloatTensor(x)
# y = torch.LongTensor(y) # 원핫이 필요없다 LongTensorfh 바꿔준다



x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=123, stratify=y)


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) 



x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_train = torch.LongTensor(y_train).to(DEVICE)
y_test = torch.LongTensor(y_test).to(DEVICE)

# x_train = torch.FloatTensor(x_train).to(DEVICE)
# x_test = torch.FloatTensor(x_test).to(DEVICE)
# y_train = torch.FloatTensor(y_train).to(DEVICE)
# y_test = torch.FloatTensor(y_test).to(DEVICE)

# x_train = torch.FloatTensor(x_train).to(DEVICE)
# x_test = torch.FloatTensor(x_test).to(DEVICE)
# y_train = torch.LongTensor(y_train).unsqueeze(-1).to(DEVICE)
# y_test = torch.LongTensor(y_test).unsqueeze(-1).to(DEVICE)


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
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 10), # y의 유니크 값이 3개라서
    nn.Softmax()      # softmax를 안해도 된다
).to(DEVICE)

criterion = nn.CrossEntropyLoss() # CrossEntropyLoss 고유의 라벨값을 알아서 맞춰준다

optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, x_train, y_train) :
    # model.train() 안써도 괜찮다
    optimizer.zero_grad() # 처음에 반영되는 가중치를 제거하기 위해 ??
    hypothesis = model(x_train)
    loss = criterion(hypothesis, y_train)
    
    loss.backward()  # 역전파를 실행한다
    optimizer.step() # 가중치를 갱신한다 음수 기울기일때 : w = w - lr * w.grad, 양수 기울기일때 : w = w + lr * w.grad
    
    return loss.item() # train() 에서의 1epochs

EPOCHS = 100
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
y_predict = torch.argmax(y_predict, axis=1)
# y_predict = y_predict.indices
print(y_predict.float())
print(y_test.float())


score = (y_predict == y_test).float().mean() # 0, 1 개수 가지고 평균을 낸것이 accuracy
print('accuracy : {:.4f}'.format(score))

from sklearn.metrics import accuracy_score
# score = accuracy_score(y_test, y_predict) # gpu상태니까 cpu로 바꿔야 한다
score = accuracy_score(y_test.cpu(), y_predict.cpu())
# score = accuracy_score(y_test.cpu().numpy(), y_predict.cpu().numpy())
print('accuracy :', score) 
# accuracy : 0.9648
# accuracy : 0.9648148148148148
