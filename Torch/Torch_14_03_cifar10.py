from calendar import c
from torchvision.datasets import CIFAR10
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import torchvision.transforms as tr
#1. Data
transf = tr.Compose([tr.Resize(15), tr.ToTensor()]) 



USE_CUDA = torch.cuda.is_available
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')

path = './_data/torch_data'

train_dataset = CIFAR10(path, train=True, download=True) # download=False
test_dataset = CIFAR10(path, train=False, download=False) # download=False

x_train, y_train = train_dataset.data/255., train_dataset.targets
x_test, y_test = test_dataset.data/255, test_dataset.targets

# print(x_train.shape) # (50000, 32, 32, 3)
# print(x_test.shape)  # (10000, 32, 32, 3)
# print(type(x_train))

x_train, x_test = x_train.reshape(-1, 32*32*3), x_test.reshape(-1, 32*32*3) # reshape view 동일
print(x_train.shape, x_test.shape) # torch.Size([60000, 784]) torch.Size([10000, 784])

x_train = torch.FloatTensor(x_train)
x_test = torch.FloatTensor(x_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

print(x_train.size())
print(x_test.size())
print(y_train.size())
print(y_test.size())


train_dset = TensorDataset(x_train, y_train)
test_dset = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_dset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dset, batch_size=32, shuffle=False) 

#2. Model
class DNN(nn.Module) :
    def __init__(self, num_feature) :
        super().__init__()
        
        self.hidden_layer1 = nn.Sequential(
            nn.Linear(num_feature, 100),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.hidden_layer2 = nn.Sequential(
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.hidden_layer3 = nn.Sequential(
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.hidden_layer4 = nn.Sequential(
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.hidden_layer5 = nn.Sequential(
            nn.Linear(100, 100),
            nn.ReLU()
        )
        self.output_layer = nn.Linear(100, 10)
        
    def forward(self, x) :
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = self.hidden_layer3(x)
        x = self.hidden_layer4(x)
        x = self.hidden_layer5(x)
        x = self.output_layer(x)
        return x
    
model = DNN(32*32*3).to(DEVICE)


#3. Compile
criterion = nn.CrossEntropyLoss()     

optimizer = optim.Adam(model.parameters(), lr=1e-4)

def train(model, criterion, optimizer , loader) :
    epoch_loss = 0
    epoch_acc = 0
    for x_batch, y_batch in loader :
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        
        optimizer.zero_grad()
        hypothesis = model(x_batch)
        loss = criterion(hypothesis, y_batch)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
        y_predict = torch.argmax(hypothesis, 1)
        acc = (y_predict == y_batch).float().mean() # True, False형태로 나오는 걸 float로 변환에서의 평균은 acc
        epoch_acc += acc.item()
        
    return epoch_loss / len(loader), epoch_acc / len(loader)
#   hist = model.fit(x_train, y_train)        hist에는 loss와 acc가 들어간다


def evaluate(model, criterion, loader) :
    model.eval() # 레이어 단계에서 배치놈 드롭아웃 등 미적용
    epoch_loss = 0
    epoch_acc = 0
    
    with torch.no_grad() :
        for x_batch, y_batch in loader :
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        
            hypothesis = model(x_batch)
            loss = criterion(hypothesis, y_batch)
            
            epoch_loss += loss.item()
            
            y_predict = torch.argmax(hypothesis, 1)
            acc = (y_predict == y_batch).float().mean()
            
            epoch_acc += acc.item()
            
        return epoch_loss / len(loader), epoch_acc / len(loader)
# loss, acc = model.evaluate(x_test, y_test)        

epochs = 20

for epoch in range(1 + epochs + 1) :
    loss, acc = train(model, criterion, optimizer, train_loader)
    
    val_loss, val_acc = evaluate(model, criterion, test_loader)
    
    print('epoch : {}, loss : {:.4f}, acc : {:.3f}, val_loss : {:.4f}, val_acc : {:.3f}'.format(epoch, loss, acc, val_loss, val_acc))
        
# epoch : 21, loss : 1.3985, acc : 0.495, val_loss : 1.4581, val_acc : 0.478

