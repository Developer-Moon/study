from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch


USE_CUDA = torch.cuda.is_available
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')


#1. Data
datasets = load_iris()
x = datasets.data
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=123, stratify=y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) 

x_train = torch.FloatTensor(x_train).to(DEVICE) 
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_train = torch.LongTensor(y_train).to(DEVICE)
y_test = torch.LongTensor(y_test).to(DEVICE)

from torch.utils.data import TensorDataset, DataLoader
train_set = TensorDataset(x_train, y_train)
test_set = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_set, batch_size=40, shuffle=True)
test_loader = DataLoader(test_set, batch_size=40, shuffle=True)


#2. Model
class Model(nn.Module) :
    def __init__(self, input_dim, output_dim) :
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 16)
        self.linear4 = nn.Linear(16, output_dim)
        self.softmax = nn.Softmax()
        self.relu = nn.ReLU()
    
    def forward(self, input_size) :
        x = self.linear1(input_size)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        x = self.softmax(x)
        return x
    
model = Model(4, 3).to(DEVICE)
    
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, loader) :
    total_loss = 0
    
    for x_batch, y_batch in loader : 
        optimizer.zero_grad()
        hypothesis = model(x_batch)
        loss = criterion(hypothesis, y_batch)
        
        loss.backward()  
        optimizer.step() 
        total_loss += loss.item()
        
    return total_loss / len(loader)

EPOCHS = 100
for epochs in range(1, EPOCHS + 1) : 
    loss = train(model, criterion, optimizer, train_loader)
    print('epochs : {}, loss : {:.8f}'.format(epochs, loss))
                                                            
print("======================= 평가, 예측 ======================= ")    
def evaluate(model, criterion, loader) :
    model.eval()
    total_loss = 0
    
    for x_batch, y_batch in loader :
        with torch.no_grad() :
            hypothesis = model(x_batch)
            loss = criterion(hypothesis, y_batch)
            total_loss += loss.item()
            
    return total_loss
    
loss = evaluate(model, criterion, test_loader)
print('loss :', loss)

y_predict = model(x_test)
y_predict = torch.argmax(y_predict, axis=1)
print(y_predict.float())
print(y_test.float())


score = (y_predict == y_test).float().mean() 
print('accuracy : {:.4f}'.format(score))

from sklearn.metrics import accuracy_score
score = accuracy_score(y_test.cpu(), y_predict.cpu())
print('accuracy :', score) 