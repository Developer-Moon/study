import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint

def preprocess(features):
    # YOUR CODE HERE
    image, label = tf.cast(features['image'], tf.float32) / 255.0, tf.one_hot(features['label'], 2)
    image = tf.image.resize(image, size=(224, 224))
    return image, label

dataset_name = 'cats_vs_dogs'
train_dataset, info = tfds.load(name=dataset_name, split="train[:20000]", with_info=True)
valid_dataset = tfds.load(name=dataset_name, split="train[20000:]")

train_dataset = train_dataset.repeat().map(preprocess).batch(32)
valid_dataset = valid_dataset.repeat().map(preprocess).batch(32)

total_size = 20000
steps_per_epoch = total_size // 32 + 1

total_valid_size = 3262
validation_steps = total_valid_size // 32 + 1


def solution_model():
    # model = # YOUR CODE HERE, BUT MAKE SURE YOUR LAST LAYER HAS 2 NEURONS ACTIVATED BY SOFTMAX
    #     tf.keras.layers.Dense(2, activation='softmax')
    # ])
    model = Sequential([
        Conv2D(64, (3, 3), input_shape=(224, 224, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(16, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dropout(0.5),
        Dense(512, activation='relu'),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid'),
    ])

    model.compile(optimizer=tf.keras.optimizers.RMSprop(0.00005), loss='binary_crossentropy',
                  metrics=['acc'])

    checkpoint_path = 'cats_dogs_checkpoint_0624.ckpt'

    checkpoint = ModelCheckpoint(checkpoint_path,
                                 save_best_only=True,
                                 save_weights_only=True,
                                 monitor='val_loss',
                                 verbose=1,
                                 )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                     mode='min',
                                                     patience=5,
                                                     factor=0.8)

    earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=10,
                                                     verbose=1)

    model.fit(train_dataset,
              steps_per_epoch=steps_per_epoch,
              epochs=10,
              validation_data=(valid_dataset),
              validation_steps=validation_steps,
              callbacks=[checkpoint, reduce_lr, earlystopping],
              verbose=1
              )

    model.load_weights(checkpoint_path)


    return model

if __name__ == '__main__':
    model = solution_model()
    model.save("model/cats_dogs_model_0524.h5")