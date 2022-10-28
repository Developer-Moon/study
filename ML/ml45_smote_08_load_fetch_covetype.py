from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import pickle

# 실습
# 증폭한 후 저장한 데이터 불러와서 
# 완성 및 성능 비교
x_train = pickle.load(open('D:\study_data\_save\_xg/ml45_smote_07_x_train.pkl', 'rb'))
y_train = pickle.load(open('D:\study_data\_save\_xg/ml45_smote_07_y_train.pkl', 'rb'))
x_test = pickle.load(open('D:\study_data\_save\_xg/ml45_smote_07_x_test.pkl', 'rb'))
y_test = pickle.load(open('D:\study_data\_save\_xg/ml45_smote_07_y_test.pkl', 'rb'))

#2. 모델구성
model = RandomForestClassifier()


#3. 훈련
model.fit(x_train, y_train)


#4. 평가, 예측
y_predict = model.predict(x_test)
score = model.score(x_test, y_test)

from sklearn.metrics import accuracy_score, f1_score

print('acc_score :', accuracy_score(y_test, y_predict))
print('f1_score(macro) :', f1_score(y_test, y_predict, average='macro')) 

# acc_score : 0.9580819772295035
# f1_score(macro) : 0.9347706177884273