#================================== One Hot Encoding이란? ==================================#
# 원핫인코딩은 인덱스에 1값을 부여하고 나머지 인덱스에 0을 부여하는 표현 방식임
# pandas에서는 get_dummies, tensorflow.keras에서는 to_categorical, 
# sklearn에서는 OneHotEncoder 함수 사용
# ==========================================================================================# 


#=============================== 1. pandas의 get_dummies ===================================#
#  - get_dummies()는 pandas의 내장함수이니만큼 pandas의 Series나 DataFrame 등에서 사용하기 편리
#  - get_dummies()는 즉각적으로 값을 변환하여 주기 때문에 직관적 
#  - pandas.get_dummies() 함수 안에 본인이 변환하고자 dataframe을 넣어주면 됨
# 

# 
#  - 주의해야할 것은 pandas가 isinstance()라는 함수를 활용해서 string일 때에 모두 더미화시킴
#    즉, 주어진 수치형 자료가 string 형태로 들어가있지 않도록 주의해야 함
#    missing_value가 있는 경우에도 get_dummies()는 문제가 생길 수 있음
#    가장 큰 단점은 새로운 데이터셋에서 작업을 할 때마다 모든 차원과 ordering이 바뀐다는 점
#===========================================================================================#  


#============================ 2. tensorflow의 to_categorical ===============================#
# tensorflow의 to_categorical은 array 배열에서 0부터 채워줌
# 예를 들어 [1, 2, 3, 4, 5] 일 경우 [0, 1, 2, 3, 4, 5]로 채워줌
#===========================================================================================# 


#=============================== 3. sklearn의 OneHotEncoder ================================
# - sklearn에서는 OneHotEncoder를 지원함
# - get_dummies()에 비해 복잡하게 느껴짐
# - 익숙한 dataframe에 즉각적으로 표현이 잘 되지 않을 뿐 아니라 비교적 생소한 
#   sparse matrix나 array 형태로 값을 반환 
# - 그러나 다양한 옵션들을 활용하면 sklearn의 단점으로 여겨지는 것들을 쉽게 해결할 수 있음
# 
#    from sklearn.preprocessing import OneHotEncoder
#    onehot_encoder = OneHotEncoder(categories='auto', sparse=False)
#    y = y.reshape(-1, 1)
#    onehot_encoder.fit(y)
#    y = onehot_encoder.transform(y)
#===========================================================================================# 

# 출처 : https://haehwan.github.io/posts/sta-Encoding/
#        https://injo.tistory.com/11
#        https://excelsior-cjh.tistory.com/175
#        https://stackabuse.com/one-hot-encoding-in-python-with-pandas-and-scikit-learn/