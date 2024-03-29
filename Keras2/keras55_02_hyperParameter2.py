from tabnanny import verbose
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Input, Dropout
import keras
from keras.optimizers import Adam

# 하이퍼 파라미터에 노드 추가, learning_rate추가


# 1. data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28*28).astype('float32')/255.
x_test = x_test.reshape(10000, 28*28).astype('float32')/255.

# from tensorflow.keras.utils import to_categorical
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

# 2. model
def build_model(drop=0.5, optimizer='adam', activation='relu', nodes=256, lr=0.01) :
    inputs = Input(shape=(28*28), name='input')
    x = Dense(nodes, activation=activation, name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(nodes, activation=activation, name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(nodes, activation=activation, name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name='outputs')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    optimizer = Adam(lr=lr)
    
    model.compile(optimizer=optimizer, metrics=['acc'], loss='sparse_categorical_crossentropy')
    
    
    return model

def create_hyperparameter():
    batchs = [100, 200, 300, 400, 500]
    optimizers = ['adam', 'rmsprop', 'adadelta']
    dropout = [0.3, 0.4, 0.5]
    activation = ['relu', 'linear', 'sigmoid', 'selu', 'elu']
    nodes = [64, 128, 256]
    lr = [0.0001, 0.001, 0.01, 0.1]
    
    return {'batch_size':batchs, 'optimizer':optimizers, 
            'drop':dropout, 'activation':activation, 'nodes':nodes, 'lr':lr}
    
hyperparameters = create_hyperparameter()


from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import time

keras_model = KerasClassifier(build_fn=build_model, verbose=1)

model = RandomizedSearchCV(keras_model, hyperparameters, cv=3, n_iter=10)

start = time.time()
model.fit(x_train, y_train, epochs=10, validation_split=0.2)
end = time.time()-start

print('걸린 시간: ', end)
print('model.best_params: ', model.best_params_)
print('model.best_estimator: ', model.best_estimator_)
print('model.best_score: ', model.best_score_)
print('model.score: ', model.score)

from sklearn.metrics import accuracy_score
y_pred = model.predict(x_test)
print('acc score: ', accuracy_score(y_test, y_pred))

# 걸린 시간:  275.9897768497467
# model.best_params:  {'optimizer': 'rmsprop', 'nodes': 128, 'lr': 0.001, 'drop': 0.3, 'batch_size': 200, 'activation': 'relu'}
# model.best_estimator:  <keras.wrappers.scikit_learn.KerasClassifier object at 0x00000245454083A0>
# model.best_score:  0.9672833482424418
# model.score:  <bound method BaseSearchCV.score of RandomizedSearchCV(cv=3,
#                    estimator=<keras.wrappers.scikit_learn.KerasClassifier object at 0x00000245200E8040>,
#                    param_distributions={'activation': ['relu', 'linear',
#                                                        'sigmoid', 'selu',
#                                                        'elu'],
#                                         'batch_size': [100, 200, 300, 400, 500],
#                                         'drop': [0.3, 0.4, 0.5],
#                                         'lr': [0.0001, 0.001, 0.01, 0.1],
#                                         'nodes': [64, 128, 256],
#                                         'optimizer': ['adam', 'rmsprop',
#                                                       'adadelta']})>
# 313/313 [==============================] - 0s 1ms/step
# acc score:  0.9767

