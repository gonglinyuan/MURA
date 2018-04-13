import os
import time

import keras.backend.tensorflow_backend as K
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard, CSVLogger
from keras.preprocessing.image import ImageDataGenerator

import dense_net_121
import hyperparameters

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

h_params = hyperparameters.load_hyperparameters()

data_train_path = '../MURA_trainval_keras/'
data_valid_path = '../MURA_valid1_keras/'
pretrained_weights_path = '../pretrain_network_weights/densenet121_weights_tf.h5'

timestamp = int(time.time())
model_save_path = '../saved_models_' + str(timestamp) + '/'
tensorboard_path = '../tensorboard_' + str(timestamp) + '/'
os.makedirs(model_save_path, mode=0o700, exist_ok=True)
os.makedirs(tensorboard_path, mode=0o700, exist_ok=True)

data_train_generator = ImageDataGenerator(horizontal_flip=True)

data_train_generator = data_train_generator.flow_from_directory(
    data_train_path,
    target_size=(256, 256),
    color_mode='rgb',
    class_mode='binary',
    batch_size=h_params['batch_size'],
    shuffle=True
)

data_valid_generator = ImageDataGenerator(horizontal_flip=True)

data_valid_generator = data_valid_generator.flow_from_directory(
    data_valid_path,
    target_size=(256, 256),
    color_mode='rgb',
    class_mode='binary',
    batch_size=h_params['batch_size'],
    shuffle=True
)

model = dense_net_121.dense_net_121(
    256, 256, color_type=3,
    weights_path=pretrained_weights_path,
    dropout_rate=h_params['dropout_dense'],
    weight_decay=h_params['weight_decay'],
    dropout_fc=h_params['dropout_fc'],
    freeze_dense_block_num=h_params['freeze_dense_block_num']
)

print(model.to_json())

model.fit_generator(
    data_train_generator,
    epochs=100,
    verbose=2,
    validation_data=data_valid_generator,
    callbacks=[
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, mode='min'),
        ModelCheckpoint(model_save_path + "dense_net_121_weights.{epoch:03d}-{val_loss:.2f}.h5",
                        monitor='val_loss', save_best_only=False, save_weights_only=True, mode='min', period=5),
        TensorBoard(log_dir=tensorboard_path),
        CSVLogger('training_' + str(timestamp) + '.log')
    ]
)

model.save(model_save_path + 'dense_net_121.h5')
model.save_weights(model_save_path + 'dense_net_121_weights.h5')
