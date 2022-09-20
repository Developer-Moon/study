from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm_notebook
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd


USE_CUDA = torch.cuda.is_available
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu') # ['cuda:0', 'cuda:1'] 2개 이상일때는 list
# print('torch :', torch.__version__, ' 사용DEVICE :', DEVICE) # torch : 1.12.1  사용DEVICE : cuda:0


# 1. Data
path = './_data/kaggle_house/'
train_set = pd.read_csv(path + 'train.csv', index_col=0)
test_set = pd.read_csv(path + 'test.csv', index_col=0)

drop_cols = ['Alley', 'PoolQC', 'Fence', 'MiscFeature'] 

test_set.drop(drop_cols, axis = 1, inplace =True)
train_set.drop(drop_cols, axis = 1, inplace =True)

cols = ['MSZoning', 'Street','LandContour','Neighborhood','Condition1','Condition2',
                'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Foundation',
                'Heating','GarageType','SaleType','SaleCondition','ExterQual','ExterCond','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',
                'BsmtFinType2','HeatingQC','CentralAir','Electrical','KitchenQual','Functional',
                'FireplaceQu','GarageFinish','GarageQual','GarageCond','PavedDrive','LotShape',
                'Utilities','LandSlope','BldgType','HouseStyle','LotConfig']

for col in tqdm_notebook(cols):
    le = LabelEncoder()
    train_set[col]=le.fit_transform(train_set[col])
    test_set[col]=le.fit_transform(test_set[col])

train_set = train_set.fillna(train_set.mean()) 
test_set = test_set.fillna(test_set.mean())

x = train_set.drop(['SalePrice'], axis=1)
y = train_set['SalePrice']
# x = torch.FloatTensor(x)
# y = torch.LongTensor(y) # 원핫이 필요없다 LongTensorfh 바꿔준다



x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, shuffle=True, random_state=1234)

print(x_train.shape) # (105, 4)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) 

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_train = torch.FloatTensor(y_train.values).unsqueeze(-1).to(DEVICE)
y_test = torch.FloatTensor(y_test.values).unsqueeze(-1).to(DEVICE)


print(x_train.shape)
print(x_test.shape) # torch.Size([171, 30, 1])
print(y_train.shape)
print(y_test.shape)


from torch.utils.data import TensorDataset, DataLoader
train_set = TensorDataset(x_train , y_train)
test_set = TensorDataset(x_test , y_test)

train_loader = DataLoader(train_set, batch_size=40, shuffle=True)
test_loader = DataLoader(test_set, batch_size=40) # 테스트 데이터는 훈련할때 영향을 끼치지 않아서 의미가 없다

class Model(nn.Module) :                         # Model class를 정의하고 nn.Module(안에 있는 변수들)을 상속하겠다 ()안의 자리는 상위 클래스만 가능하다
    def __init__(self, input_dim, output_dim) :  # init 정의 단계 - 클래스 안에는 __init__라는 함수(생성자)가 들어간다 - 정의 하는 순간 실행된다 input_dim은 매개변수
        # super().__init__()                     # super - nn.Module(아빠)의 생성자까지 다 쓰겠다(정의 하지 않으면 에러 발생)
        super(Model, self).__init__()            # 위에꺼와 같은 표현
        self.linear1 = nn.Linear(input_dim, 64)  # self - 이 클래스 안에서 쓸꺼다
        self.linear2 = nn.Linear(64, 32)        
        self.linear3 = nn.Linear(32, 16)
        self.linear4 = nn.Linear(16, output_dim)
        self.relu = nn.ReLU()
        # self.softmax = nn.Softmax()             # init과 forward 는 nn.Module을 상속 받는다
        
    def forward(self, input_size) :              # 실행 단계 - 모델구성
        x = self.linear1(input_size)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        return x
    
model = Model(75, 1).to(DEVICE)

criterion = nn.MSELoss() # criterion = Loss, 

optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, loader) :
    # model.train() 안써도 괜찮다
    total_loss = 0
    for x_batch, y_batch in loader :
        optimizer.zero_grad() # 처음에 반영되는 가중치를 제거하기 위해 ??
        hypothesis = model(x_batch)
        loss = criterion(hypothesis, y_batch)
        loss.backward()  # 역전파를 실행한다
        optimizer.step() # 가중치를 갱신한다 음수 기울기일때 : w = w - lr * w.grad, 양수 기울기일때 : w = w + lr * w.grad
        total_loss += loss.item()
    
    return total_loss / len(loader)

EPOCHS = 500
for epochs in range(1, EPOCHS + 1) : 
    loss = train(model, criterion, optimizer, train_loader)
    print('epochs : {}, loss : {:.8f}'.format(epochs, loss)) # 정규표현식 '' {} 메모리 값을 불러오는 공간을 만든 것 그걸 불러 오는 곳이 .foramt(epochs, loss) format안의 순서대로 
                                                             # 
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
# r2_score : 0.8263144765000853

# r2_score : 0.6982788370500305