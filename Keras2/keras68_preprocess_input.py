from keras.applications.resnet import preprocess_input, decode_predictions
from keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
import numpy as np


model = ResNet50(weights='imagenet') # weights를 imagenet에서 가져온거다 / 이미지넷 대회에서 쓴 오브젝트 분류 데이터를 끌어옴
img_path = 'D:/study_data/_data/dog/123.jpg'
img = image.load_img(img_path, target_size=(224, 224))
print(img) # <PIL.Image.Image image mode=RGB size=224x224 at 0x198C6218610>

x = image.img_to_array(img)
print("============================= image.img_to_array(img) =============================")
print(x, '\n', x.shape) # (224, 224, 3)

x = np.expand_dims(x, axis=0) # expand_dims : 차원을 늘린다  axis= 늘려주고싶은 지점을 지정
print("============================= np.expand_dims(x, axis=0) =============================")
print(x, '\n', x.shape) # axis=0 (1, 224, 224, 3)
 
print(np.min(x), np.max(x)) # 0.0 255.0
x = preprocess_input(x) # 0~255 데이터에서 -150~150 데이터로 변경                       
print("============================= x = preprocess_input(x) =============================")
print(x, '\n', x.shape)
print(np.min(x), np.max(x)) # -123.68 151.061



pred = model.predict(x)
print(pred, '\n', pred.shape)  # (1, 1000)
print(np.argmax(pred, axis=1)) # [254]

print('결과는 :', decode_predictions(pred, top=5)[0]) # 상위 5개
# 결과는 : [('n02110958', 'pug', 0.8825372),
#        ('n02112706', 'Brabancon_griffon', 0.08520029),
#        ('n02093256', 'Staffordshire_bullterrier', 0.020956889),
#        ('n02108915', 'French_bulldog', 0.0058413814),
#        ('n02108422', 'bull_mastiff', 0.0021840474)]

