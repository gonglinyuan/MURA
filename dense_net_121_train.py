import functools
import numpy as np
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from PIL import Image
import dense_net_121
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

data_train_path = '../MURA_trainval_keras/'
data_valid_path = '../MURA_valid1_keras/'
pretrained_weights_path = '../pretrain_network_weights/densenet121_weights_tf.h5'
model_save_path = '../saved_models/'

data_train_generator = ImageDataGenerator(horizontal_flip=True)

data_train_generator = data_train_generator.flow_from_directory(
    data_train_path,
    target_size=(256, 256),
    color_mode='rgb',
    class_mode='binary',
    batch_size=16,
    shuffle=True
)

data_valid_generator = ImageDataGenerator(horizontal_flip=True)

data_valid_generator = data_valid_generator.flow_from_directory(
    data_valid_path,
    target_size=(256, 256),
    color_mode='rgb',
    class_mode='binary',
    batch_size=16,
    shuffle=True
)

model = dense_net_121.dense_net_121(256, 256, color_type=3, weights_path=pretrained_weights_path)

print(model.to_json())

K.tensorflow_backend._get_available_gpus()

model.fit_generator(
    data_train_generator,
    epochs=100,
    verbose=2,
    validation_data=data_valid_generator,
    callbacks=[
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, mode='min'),
        ModelCheckpoint("dense_net_121_weights.{epoch:03d}-{val_loss:.2f}.h5", monitor='val_loss', save_best_only=False, save_weights_only=True, mode='min', period=10)
    ]
)

model.save(model_save_path + 'dense_net_121.h5')
model.save_weights(model_save_path + 'dense_net_121_weights.h5')
