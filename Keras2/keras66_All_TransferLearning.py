from keras.applications import VGG16, VGG19
from keras.applications import ResNet50, ResNet50V2
from keras.applications import ResNet101, ResNet101V2, ResNet152, ResNet152V2
from keras.applications import DenseNet121, DenseNet169, DenseNet201
from keras.applications import InceptionV3, InceptionResNetV2
from keras.applications import MobileNet, MobileNetV2
from keras.applications import MobileNetV3Small, MobileNetV3Large
from keras.applications import NASNetLarge, NASNetMobile
from keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB7
from keras.applications import Xception
from keras.layers import Dense, Flatten
from keras.models import Sequential


#2. 모델구성
models = [VGG16, VGG19, ResNet50, ResNet50V2, ResNet101, ResNet101V2, ResNet152, ResNet152V2, DenseNet121, DenseNet169, DenseNet201,
          InceptionV3, InceptionResNetV2, MobileNet, MobileNetV2, MobileNetV3Small, MobileNetV3Large, NASNetLarge, NASNetMobile,
          EfficientNetB0, EfficientNetB1, EfficientNetB7, Xception]

for i in models :
    target = i()
    # target.trainable=False # 가중치 동결

    model = Sequential()
    model.add(target)
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    
    print("=========================")
    print("모델명 :", target.name)
    print("전체 가중치 갯수 :", len(model.weights))
    print("훈련 가능 가중치 갯수 :", len(model.trainable_weights))

#
# 모델명 : vgg16
# 전체 가중치 갯수 : 34
# 훈련 가능 가중치 갯수 : 34
# =========================
# 모델명 : vgg19
# 전체 가중치 갯수 : 40
# 훈련 가능 가중치 갯수 : 40
# =========================
# 모델명 : resnet50
# 전체 가중치 갯수 : 322
# =========================
# 모델명 : resnet50v2
# 전체 가중치 갯수 : 274
# 훈련 가능 가중치 갯수 : 176
# =========================
# 모델명 : resnet101
# 전체 가중치 갯수 : 628
# 훈련 가능 가중치 갯수 : 420
# =========================
# 모델명 : resnet101v2
# 전체 가중치 갯수 : 546
# 훈련 가능 가중치 갯수 : 346
# =========================
# 모델명 : resnet152
# 전체 가중치 갯수 : 934
# 훈련 가능 가중치 갯수 : 624
# =========================
# 모델명 : resnet152v2
# 전체 가중치 갯수 : 818
# 훈련 가능 가중치 갯수 : 516
# =========================
# 모델명 : densenet121
# 전체 가중치 갯수 : 608
# 훈련 가능 가중치 갯수 : 366
# =========================
# 모델명 : densenet169
# 전체 가중치 갯수 : 848
# 훈련 가능 가중치 갯수 : 510
# =========================
# 모델명 : densenet201
# 전체 가중치 갯수 : 1008
# 훈련 가능 가중치 갯수 : 606
# =========================
# 모델명 : inception_v3
# 전체 가중치 갯수 : 380
# 훈련 가능 가중치 갯수 : 192
# =========================
# 모델명 : inception_resnet_v2
# 전체 가중치 갯수 : 900
# 훈련 가능 가중치 갯수 : 492
# =========================
# 모델명 : mobilenet_1.00_224
# 전체 가중치 갯수 : 139
# 훈련 가능 가중치 갯수 : 85
# =========================
# 모델명 : mobilenetv2_1.00_224
# 전체 가중치 갯수 : 264
# 훈련 가능 가중치 갯수 : 160
# =========================
# 모델명 : MobilenetV3small
# 전체 가중치 갯수 : 212
# 훈련 가능 가중치 갯수 : 144
# =========================
# 모델명 : MobilenetV3large
# 전체 가중치 갯수 : 268
# 훈련 가능 가중치 갯수 : 176
# =========================
# 모델명 : NASNet
# 전체 가중치 갯수 : 1548
# 훈련 가능 가중치 갯수 : 1020
# =========================
# 모델명 : NASNet
# 전체 가중치 갯수 : 1128
# 훈련 가능 가중치 갯수 : 744
# =========================
# 모델명 : efficientnetb0
# 전체 가중치 갯수 : 316
# 훈련 가능 가중치 갯수 : 215
# =========================
# 모델명 : efficientnetb1
# 전체 가중치 갯수 : 444
# 훈련 가능 가중치 갯수 : 303
# =========================
# 모델명 : efficientnetb7
# 전체 가중치 갯수 : 1042
# 훈련 가능 가중치 갯수 : 713
# =========================
# 모델명 : xception
# 전체 가중치 갯수 : 238
# 훈련 가능 가중치 갯수 : 158



# trainable=False
# =========================
# 모델명 : vgg16
# 전체 가중치 갯수 : 34    
# 훈련 가능 가중치 갯수 : 2
# =========================
# 모델명 : vgg19
# 전체 가중치 갯수 : 40    
# 훈련 가능 가중치 갯수 : 2
# =========================
# 모델명 : resnet50        
# 전체 가중치 갯수 : 322   
# 훈련 가능 가중치 갯수 : 2
# =========================
# 모델명 : resnet50v2      
# 전체 가중치 갯수 : 274   
# 훈련 가능 가중치 갯수 : 2
# =========================
# 모델명 : resnet101       
# 전체 가중치 갯수 : 628   
# 훈련 가능 가중치 갯수 : 2
# =========================
# 모델명 : resnet101v2     
# 전체 가중치 갯수 : 546   
# 훈련 가능 가중치 갯수 : 2
# =========================
# 모델명 : resnet152       
# 전체 가중치 갯수 : 934   
# 훈련 가능 가중치 갯수 : 2
# =========================
# 모델명 : resnet152v2  
# 전체 가중치 갯수 : 818
# 훈련 가능 가중치 갯수 : 2
# =========================
# 모델명 : densenet121
# 전체 가중치 갯수 : 608
# 훈련 가능 가중치 갯수 : 2
# =========================
# 모델명 : densenet169
# 전체 가중치 갯수 : 848
# 훈련 가능 가중치 갯수 : 2
# =========================
# 모델명 : densenet201
# 전체 가중치 갯수 : 1008
# 훈련 가능 가중치 갯수 : 2
# =========================
# 모델명 : inception_v3
# 전체 가중치 갯수 : 380
# 훈련 가능 가중치 갯수 : 2
# =========================
# 모델명 : inception_resnet_v2
# 전체 가중치 갯수 : 900
# 훈련 가능 가중치 갯수 : 2
# =========================
# 모델명 : mobilenet_1.00_224
# 전체 가중치 갯수 : 139
# 훈련 가능 가중치 갯수 : 2
# =========================
# 모델명 : mobilenetv2_1.00_224
# 전체 가중치 갯수 : 264
# 훈련 가능 가중치 갯수 : 2
# =========================
# 모델명 : MobilenetV3small
# 전체 가중치 갯수 : 212
# 훈련 가능 가중치 갯수 : 2
# =========================
# 모델명 : MobilenetV3large
# 전체 가중치 갯수 : 268
# 훈련 가능 가중치 갯수 : 2
# =========================
# 모델명 : NASNet
# 전체 가중치 갯수 : 1548
# 훈련 가능 가중치 갯수 : 2
# =========================
# 모델명 : NASNet
# 전체 가중치 갯수 : 1128
# 훈련 가능 가중치 갯수 : 2
# =========================
# 모델명 : efficientnetb0
# 전체 가중치 갯수 : 316
# 훈련 가능 가중치 갯수 : 2
# =========================
# 모델명 : efficientnetb1
# 전체 가중치 갯수 : 444
# 훈련 가능 가중치 갯수 : 2
# =========================
# 모델명 : efficientnetb7
# 전체 가중치 갯수 : 1042
# 훈련 가능 가중치 갯수 : 2
# =========================
# 모델명 : xception
# 전체 가중치 갯수 : 238
# 훈련 가능 가중치 갯수 : 2