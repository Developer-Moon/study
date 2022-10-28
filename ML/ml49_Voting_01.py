from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd 


#1. 데이터 
datasets = load_breast_cancer()
df = pd.DataFrame(datasets.data, columns=datasets.feature_names) # 컬럼까지 데이터프레임화
print(df.head(7)) # 7개 나온다

x_train, x_test, y_train, y_test = train_test_split(datasets.data, datasets.target, train_size=0.8, random_state=123, stratify=datasets.target)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)



#2. 모델구성
lr = LogisticRegression()
knn = KNeighborsClassifier(n_neighbors=8)
model = VotingClassifier(estimators=[('LR', lr), ('knn', knn)], voting='soft') # voting:'hard'



#3. 훈련
model.fit(x_train, y_train)



#4. 평가, 예측
y_predict = model.predict(x_test)
score = accuracy_score(y_test, y_predict)
print('보팅결과 :', round(score, 4))  

classifiers =[lr, knn]
for model2 in classifiers:
    model2.fit(x_train, y_train)
    y_predict = model2.predict(x_test)
    score2 = accuracy_score(y_test, y_predict)
    class_name = model2.__class__.__name__ # 모델의 이름을 반환
    print('{0} 정확도 : {1:.4f}'.format(class_name, score2))

# 보팅결과 : 0.9912
# LogisticRegression 정확도 : 0.9737
# KNeighborsClassifier 정확도 : 0.9912



'''
하드보팅
        a  b  c
이진    0  0  1  -> 0
0,1     1  1  0  -> 1
소프트보팅
        a       b        c
이진   0 1     0 1      0 1
0,1  0.7 0.3  0.5 0.5  0.6 0.4
0: (0.7+0.5+0.6)/3
1: (0.3+0.5+0.4)/3
-> 이건 (0.7+0.5+0.6)/3의 확률로 0이다
'''