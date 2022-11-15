import numpy as np          
from keras.preprocessing.image import ImageDataGenerator


train_datagen = ImageDataGenerator(  # ImageDataGenerator - 이미지를 숫자화 이 옵션들은 증폭작업에 사용 (임의로 이중 1개만 랜덤으로 적용된다)
    rescale=1./225,                  # 스케일링 자체 minmax제공
    horizontal_flip=True,            # 수평 방향 뒤집기
    vertical_flip=True,              # 수직 방향 뒤집기 
    width_shift_range=0.1,           # 좌우이동
    height_shift_range=0.1,          # 상하이동
    rotation_range=5,                # 돌리기
    zoom_range=1.2,                  # 확대
    shear_range=0.7,                 # 찌그러 뜨리는거
    fill_mode='nearest'              # 이미지를 회전, 이동하거나 축소할 때 생기는 공간을 채우는 방식
)

test_datagen = ImageDataGenerator(rescale=1./255) # 평가 데이터는 증폭시키지 않는다, 확인해야하는 값은 그대로 사용하기 떄문 

xy_train = train_datagen.flow_from_directory(     # flow_from_directory - 폴더(directory)에서 가져온다
        'D:/study_data/_data/image/brain/train/',
        target_size=(150, 150),                   # 수집한 image의 크기가 다 다르니까 크기를 일정하게 해준다 
        batch_size=5,                             # 만약 이미지 개수가 안나눠떨어지면 뒤에 몇장의 이미지는 가져오지 못하고 버리는 셈
        class_mode='binary',                      # binary(이진분류)
        shuffle=True,
        color_mode='grayscale'  # 디폴트값은 컬러
        # Found 160 images belonging to 2 classes.
    )

xy_test = test_datagen.flow_from_directory(                
        'D:/study_data/_data/image/brain/test/',
        target_size=(150, 150), 
        batch_size=5,
        class_mode='binary',                                
        shuffle=True,
        color_mode='grayscale'
        # Found 160 images belonging to 2 classes.
    )

print(xy_train)
# <keras.preprocessing.image.DirectoryIterator object at 0x0000017DD2D74F70>
# sklearn 데이터형식과 같음 ex)load_boston()처럼
# from sklearn.datasets import load_boston
# datasets = load_boston()
# print(datasets)

# 160개의 데이터가 배치 5개 단위로 짤렸고 5개 단위로 잘린게 32개다 
# print(xy_train[33]) - Error

print(xy_train[0])   # xy와 다 같이 들어있다

print(xy_train[0][0])       # 마지막 배치
print(xy_train[0][0].shape) # x만 나온다
print(xy_train[0][1]) 
# print(xy_train[31][2]) # 에러 0과 1만 나오니까


print(xy_train[0][0].shape, xy_train[0][1].shape)


print(type(xy_train))       # <class 'keras.preprocessing.image.DirectoryIterator'> Iterator 반복자 
print(type(xy_train[0]))    # <class 'tuple'> 0번째에는 x, y가 들어가있다
print(type(xy_train[0][0])) # <class 'numpy.ndarray'> 0번째에는 x, y가 들어가있다
print(type(xy_train[0][1])) # <class 'numpy.ndarray'> 1번째에는 x, y가 들어가있다 배치단위로 묶여있다



















xy_train = train_datagen.flow_from_directory(
    'd:/study_data/_data/image/brain/train/', # 이 경로의 이미지파일을 불러 수치화
    target_size=(150,150),# 크기들을 일정하게 맞춰준다.
    batch_size=5,
    class_mode='binary', 
    # color_mode='grayscale', #디폴트값은 컬러
    shuffle=True,
    )#Found 160 images belonging to 2 classes

xy_test = test_datagen.flow_from_directory(
   'd:/study_data/_data/image/brain/test/', # 이 경로의 이미지파일을 불러 수치화
    target_size=(150,150),# 크기들을 일정하게 맞춰준다.
    batch_size=5,
    class_mode='binary',
    # color_mode='grayscale',
    shuffle=True,
    )#Found 120 images belonging to 2 classes.

print(xy_train)

#<keras.preprocessing.image.DirectoryIterator object at 0x000002082E821D90>
print(xy_train[0]) #0~31 batch가 5이므로 160장이므로 31장으로 나뉘어짐
print(xy_train[0][0].shape) #x값
print(xy_train[0][1].shape) #y값

print(type(xy_train)) #<class 'keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0]))#<class 'tuple'> x,y값
print(type(xy_train[0][0]))#<class 'numpy.ndarray'>
print(type(xy_train[0][1]))#<class 'numpy.ndarray'>
# image 데이터를 가져왔을 때 x numpy y numpy 형태로 batch단위로 묶여있다!