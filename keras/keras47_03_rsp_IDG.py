import numpy as np          
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


#1. 데이터
datagen = ImageDataGenerator(  # ImageDataGenerator - 이미지를 숫자화 이 옵션들은 증폭작업에 사용 (임의로 이중 1개만 랜덤으로 적용된다)
    rescale=1./225, 
    # horizontal_flip=True,   # 수평방향 뒤집기  뒤집으면 2개가 된다   
    # vertical_flip=True,     # 수직 방향 뒤집기 
    # width_shift_range=0.1,  # 좌우이동
    # height_shift_range=0.1, # 상하이동
    # rotation_range=5,       # 돌리기
    # zoom_range=1.2,         # 확대
    # shear_range=0.7,        # 이미지 기울기(확인 필요, 3차원 기울기??)
    # fill_mode='nearest'     # 이미지를 회전, 이동하거나 축소할 때 생기는 공간을 채우는 방식
)


xy = datagen.flow_from_directory(
    'D:/study_data/_data/image/rps/rps',
    target_size=(150, 150), # 수집한 image의 크기가 다 다르니까 크기를 일정하게 해준다 - 가로세로가 일정한 비율이 아니라면?
    batch_size=3000,
    class_mode='categorical',    # categorical은 one hot되서 나온다                              
    shuffle=True,
    # color_mode='color'  # 디폴트값은 컬러
)   
print(xy)
# Found 2520 images belonging to 3 classes.

np.save('d:/study_data/_save/_npy/kears_47_01_x.npy', arr=xy[0][0])  # test_x
np.save('d:/study_data/_save/_npy/kears_47_02_y.npy', arr=xy[0][1])  # test_y

