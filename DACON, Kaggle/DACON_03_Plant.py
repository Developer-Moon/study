import pandas as pd
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from xgboost import XGBClassifier, XGBRegressor
from tensorflow.python.keras.models import Sequential 
from tensorflow.python.keras.layers import Dense, GRU, LSTM, Dropout





path = './_data/dacon_plant/'
all_input_list = sorted(glob.glob(path + 'train_input/*.csv'))
all_target_list = sorted(glob.glob(path + 'train_target/*.csv'))

train_input_list = all_input_list[:50]
train_target_list = all_target_list[:50]

val_input_list = all_input_list[50:]
val_target_list = all_target_list[50:]

# print(all_input_list)
print(val_input_list)
print(len(val_input_list))  # 8

def aaa(input_paths, target_paths): #, infer_mode):
    input_paths = input_paths
    target_paths = target_paths
    # self.infer_mode = infer_mode
   
    data_list = []
    label_list = []
    print('시작...')
    # for input_path, target_path in tqdm(zip(input_paths, target_paths)):
    for input_path, target_path in zip(input_paths, target_paths):
        input_df = pd.read_csv(input_path)
        target_df = pd.read_csv(target_path)
       
        input_df = input_df.drop(columns=['시간'])
        input_df = input_df.fillna(0)
       
        input_length = int(len(input_df)/1440)
        target_length = int(len(target_df))
        print(input_length, target_length)
       
        for idx in range(target_length):
            time_series = input_df[1440*idx:1440*(idx+1)].values
            # self.data_list.append(torch.Tensor(time_series))
            data_list.append(time_series)
        for label in target_df["rate"]:
            label_list.append(label)
    return np.array(data_list), np.array(label_list)
    print('끗.')

x_train, y_train = aaa(train_input_list, train_target_list) #, False)
x_test, y_test = aaa(val_input_list, val_target_list) #, False)


print(x_train[0])
print(len(x_train), len(y_train)) # 1607 1607
print(len(x_train[0]))   # 1440
print(y_train)   # 1440
print(x_train.shape, y_train.shape)   # (1607, 1440, 37) (1607,)
print(x_test.shape, y_test.shape)     # (206 , 1440, 37) (206,)

# x_train = x_train.reshape(1607, 1440*37)
# x_test = x_test.reshape(206 , 1440*37)

#2. 모델구성
# model = XGBRegressor(tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0)

model = Sequential()  
# model.add(SimpleRNN(10, input_shape=(3, 1))) # [batch, timesteps, feature].
model.add(LSTM(units=100, input_shape=(1440, 37))) # input_shape가  input_length=3, input_dim=1 로 사용가능
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))

#3. 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, batch_size=150, epochs=1)



#4. 결과, 예측
result = model.evaluate(x_test, y_test)
print('model.score :', result)

y_predict = model.predict(x_test)
# r2 = r2_score(y_test, y_predict)
# print('r2_score :', r2)


# model.score : -1.1955752854294874
# r2_score : -1.1955752854294874






# y_predict -> TEST_ files
for i in range(6):
    thislen=0
    thisfile = './_data/dacon_plant/sample_submission/'+'TEST_0'+str(i+1)+'.csv' 
    test = pd.read_csv(thisfile, index_col=False)
    test['rate'] = y_predict[thislen:thislen+len(test['rate'])]
    test.to_csv(thisfile, index=False)
    thislen+=len(test['rate'])


# TEST_ files -> zip file
import zipfile
filelist = ['TEST_01.csv','TEST_02.csv','TEST_03.csv','TEST_04.csv','TEST_05.csv', 'TEST_06.csv']
os.chdir("./_data/dacon_plant/sample_submission/")
with zipfile.ZipFile("submissionKeras.zip", 'w') as my_zip:
    for i in filelist:
        my_zip.write(i)
    my_zip.close()

# 5. 제출 준비
