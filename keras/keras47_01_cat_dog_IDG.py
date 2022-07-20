import numpy as np          
from keras.preprocessing.image import ImageDataGenerator


#1. 데이터
train_datagenerator = ImageDataGenerator(rescale=1./225,)
test_datagenerator = ImageDataGenerator(rescale=1./255)

xy_train = train_datagenerator.flow_from_directory('D:/study_data/_data/image/cat_dog/training_set/',
        target_size=(150, 150), 
        batch_size=80,
        class_mode='binary',                                  
        shuffle=True,
        # color_mode='grayscale'  
    )

xy_test = test_datagenerator.flow_from_directory('D:/study_data/_data/image/cat_dog/test_set/',
        target_size=(150, 150), 
        batch_size=5,
        class_mode='binary',                                 
        shuffle=True,
        # color_mode='grayscale'
    )

print(xy_train) # Found 8005 images belonging to 1 classes.
print(xy_test)  # Found 2023 images belonging to 1 classes.

np.save('d:/study_data/_save/_npy/kears_47_01_train_x.npy', arr=xy_train[0][0])
np.save('d:/study_data/_save/_npy/kears_47_02_train_y.npy', arr=xy_train[0][1])
np.save('d:/study_data/_save/_npy/kears_47_03_test_x.npy', arr=xy_test[0][0]) 
np.save('d:/study_data/_save/_npy/kears_47_04_test_y.npy', arr=xy_test[0][1])  