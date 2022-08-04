from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict, StratifiedKFold  # Kfold - cross_val_score검증하기위해 이걸 쓴다
from sklearn.datasets import load_digits
from sklearn.metrics import r2_score, accuracy_score
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler 
import numpy as np





# 1. 데이터
path = './_data/kaggle_titanic/'
train_set = pd.read_csv(path + 'train.csv', index_col=0)
test_set = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'gender_submission.csv')
print('train.shape, test.shape, submit.shape', 
      train_set.shape, test_set.shape, submission.shape)    # (891, 11) (418, 10) (418, 2)

# 데이터 전처리

# Ticket, Cabin[선실], Name 삭제   
train_set = train_set.drop(['Ticket', 'Cabin', 'Name'], axis=1)
test_set = test_set.drop(['Ticket', 'Cabin', 'Name'], axis=1)


# Age NaN값 변환
train_set['Age'] = train_set['Age'].fillna(train_set.Age.dropna().mode()[0])
test_set['Age'] = test_set['Age'].fillna(train_set.Age.dropna().mode()[0])


# Embarked, Sex NaN값 및 Object => int 변환
train_set['Embarked'] = train_set['Embarked'].fillna(train_set.Embarked.dropna().mode()[0]) 
train_set['Embarked'] = train_set['Embarked'].map({'S':0, 'C':1, 'Q':2}).astype(int)  
test_set['Embarked'] = test_set['Embarked'].fillna(test_set.Embarked.dropna().mode()[0])
test_set['Embarked'] = test_set['Embarked'].map({'S':0, 'C':1, 'Q':2}).astype(int)

train_set['Sex'] = train_set['Sex'].fillna(train_set.Sex.dropna().mode()[0])
train_set['Sex'] = train_set['Sex'].map({'male':0, 'female':1}).astype(int)
test_set['Sex'] = test_set['Sex'].fillna(test_set.Sex.dropna().mode()[0])
test_set['Sex'] = test_set['Sex'].map({'male':0, 'female':1}).astype(int)


# x, y 데이터
x = train_set.drop(['Survived'], axis=1)
y = train_set['Survived']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, shuffle=True, random_state=3)

scaler = RobustScaler()

scaler.fit(x_train)                     
x_train = scaler.transform(x_train)    
x_test = scaler.transform(x_test)      
test_set = scaler.transform(test_set)  
                    
                             
#2. 모델구성
model = SVC()


#3. 컴파일, 훈련, 평가, 예측
scores = cross_val_score(model, x_train, y_train, cv=kfold) # cv=5 라면 kfold를 5로 쓴다
print('ACC :', scores, '\n cross_val_score :' , round(np.mean(scores), 4)) # 4번째까지 출력 (반올림을 5번째 자리에서)

y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
print(y_predict)

acc = accuracy_score(y_test, y_predict)
print('cross_val_predict ACC : ', acc)

# cross_val_predict ACC :  0.9722222222222222