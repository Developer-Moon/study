from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim



USE_CUDA = torch.cuda.is_available
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu') # ['cuda:0', 'cuda:1'] 2개 이상일때는 list


# 1. Data
datasets = load_breast_cancer()
x = datasets.data
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=123, stratify=y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) 

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_train = torch.FloatTensor(y_train).unsqueeze(-1).to(DEVICE)
y_test = torch.FloatTensor(y_test).unsqueeze(-1).to(DEVICE)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# x, y를 뭉치고 배치한다
# 데이터에서 배치를 얼마만큼 할 것인가 정하고 dataloader에서 뭉친다

####################### DataLoader #######################
from torch.utils.data import TensorDataset ,DataLoader # TensorDataset 데이터를 합치는 모듈, 배치까지 합치는 모듈 DataLoader
train_set = TensorDataset(x_train, y_train) # x, y를 합친다
test_set = TensorDataset(x_test, y_test)    # x, y를 합친다
# print(train_set) # <torch.utils.data.dataset.TensorDataset object at 0x0000012FBF1E6F10>
print('-------------train_set[0]--------------')
print(train_set[0]) # x_train 0번째 행에 대한 내용 train_set[0][0]와 train_set[0][1] 두개를 다 보여준다
print('-------------train_set[0][0]--------------')
print(train_set[0][0])
print('-------------train_set[0][1]--------------')
print(train_set[0][1]) # 
print(len(train_set))  # 398개

# x, y 배치 합체
train_loader = DataLoader(train_set, batch_size=40, shuffle=True) # 배치 사이즈 40개로 trainset을 섞어서 합친다 
test_loader = DataLoader(test_set, batch_size=40) 


#2. 모델
# model = nn.Sequential(
#     nn.Linear(30, 64),
#     nn.ReLU(),
#     nn.Linear(64, 32),#
#     nn.ReLU(),
#     nn.Linear(32, 1),
#     nn.Sigmoid()
# ).to(DEVICE)
class Model(nn.Module) :                         # Model class를 정의하고 nn.Module(안에 있는 변수들)을 상속하겠다 ()안의 자리는 상위 클래스만 가능하다
    def __init__(self, input_dim, output_dim) :  # init 정의 단계 - 클래스 안에는 __init__라는 함수(생성자)가 들어간다 - 정의 하는 순간 실행된다 input_dim은 매개변수
        # super().__init__()                     # super - nn.Module(아빠)의 생성자까지 다 쓰겠다(정의 하지 않으면 에러 발생)
        super(Model, self).__init__()            # 위에꺼와 같은 표현
        self.linear1 = nn.Linear(input_dim, 64)  # self - 이 클래스 안에서 쓸꺼다
        self.linear2 = nn.Linear(64, 32)        
        self.linear3 = nn.Linear(32, 16)
        self.linear4 = nn.Linear(16, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()              # init과 forward 는 nn.Module을 상속 받는다
        
    def forward(self, input_size) :              # 실행 단계 - 모델구성
        x = self.linear1(input_size)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        x = self.sigmoid(x)
        return x
        
model = Model(30, 1).to(DEVICE)       

criterion = nn.BCELoss()

optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, loader) :
    # model.train() 안써도 괜찮다
    
                                        # 텐서 fit에서 하는 작업
    total_loss = 0 # list를 써도 상관 X  # 지금까지는 통째로의 데이터를 작업했다 
    for x_batch, y_batch in loader :    # for문에 넣고 돌리면 batch크기로 빼준다
        optimizer.zero_grad()           # 배치 작업을 할때도 쓴다
        hypothesis = model(x_batch)     # 40개만 가지고 구한 것의 y배치와 비교하여 
        loss = criterion(hypothesis, y_batch) # 
    
        loss.backward()  # 역전파를 실행한다
        optimizer.step() # 
        total_loss += loss.item() # 40개를 1번 돌린 loss값을 total_loss에 넣고 그 다음번 째 돌린걸 넣는다 
    
    return total_loss / len(loader) # loader 개수로 나눠주면 된다 # 평균을 안해줘도 된다 다음 훈련과 비교하여 그것 과 비교만 하면 되니까 

EPOCHS = 100
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

y_predict = (model(x_test) >= 0.5).float() # (model(x_test) 이거 다시 질문 해야한다 !!!!!!!!!!!!!!!!!!!!!
print(y_predict[:10])

score = (y_predict == y_test).float().mean() # 0, 1 개수 가지고 평균을 낸것이 accuracy
print('accuracy : {:.4f}'.format(score))

from sklearn.metrics import accuracy_score
# score = accuracy_score(y_test, y_predict) # gpu상태니까 cpu로 바꿔야 한다
score = accuracy_score(y_test.cpu(), y_predict.cpu())
# score = accuracy_score(y_test.cpu().numpy(), y_predict.cpu().numpy())
print('accuracy :', score) 
# accuracy : 0.9766
# accuracy : 0.9766081871345029

path = './save'
torch.save(model.state_dict(), path + 'torch13_state_dict.pt')

