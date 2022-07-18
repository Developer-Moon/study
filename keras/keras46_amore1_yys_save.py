from tensorflow.python.keras.models import Sequential   
from tensorflow.python.keras.layers import Dense 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error   
import numpy as np                                               
import pandas as pd



#1. 데이터
amore = pd.read_csv('./_data/test_amore_0718/amore.csv', encoding = 'CP949')
samsung = pd.read_csv('./_data/test_amore_0718/samsung.csv', encoding = 'CP949')
print(amore.columns)
print(samsung.columns)
# ['일자', '시가', '고가', '저가', '종가', '전일비', 'Unnamed: 6', '등락률', '거래량', '금액(백만)', '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비']

amore.rename(columns = {'일자':'date', '시가':'open', '고가':'high', '저가':'row', '종가':'close', '거래량':'trading', '기관':'company'}, inplace=True)


amore['date'] = pd.to_datetime(amore['date'])
amore['year'] = amore['date'].dt.year
amore['month'] = amore['date'].dt.month
amore['day'] = amore['date'].dt.day


amore = amore.dropna()
amore = amore.drop(['date', '전일비', 'Unnamed: 6', '등락률', '금액(백만)', '신용비', '개인', '외인(수량)', '외국계', '프로그램', '외인비'], axis=1)
print(amore.columns)
print(amore.isnull().sum())


