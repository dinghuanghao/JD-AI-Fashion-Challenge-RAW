import math
import os
import time

import keras
import numpy as np
from keras.layers import Dense, BatchNormalization, Conv2D, MaxPooling2D, Flatten
from keras import Sequential

import config
from util import data_loader
from util import keras_util
from util import metrics
from util.keras_util import KerasModelConfig

model_config = KerasModelConfig(k_fold_file="1.txt",
                                model_path=os.path.abspath(__file__),
                                image_resolution=224,
                                data_type=[config.DATA_TYPE_SEGMENTED],
                                label_position=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                                train_batch_size=32,
                                val_batch_size=256,
                                predict_batch_size=256,
                                epoch=[15],
                                lr=[0.001],
                                freeze_layers=[0])


def get_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=model_config.image_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(13, activation='sigmoid'))

    model.compile(loss="binary_crossentropy",
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy', metrics.smooth_f2_score])
    return model


def train():
    y_train = np.array(data_loader.get_labels(model_config.train_files), np.bool)[:, model_config.label_position]
    y_valid = np.array(data_loader.get_labels(model_config.val_files), np.bool)[:, model_config.label_position]

    tensorboard = keras_util.TensorBoardCallback(log_dir=model_config.record_dir, log_every=1,
                                                 model_config=model_config)
    checkpoint = keras.callbacks.ModelCheckpoint(filepath=model_config.save_model_format,
                                                 save_weights_only=True)

    train_flow = data_loader.KerasGenerator(model_config=model_config,
                                            featurewise_center=True,
                                            featurewise_std_normalization=True,
                                            width_shift_range=0.15,
                                            height_shift_range=0.1,
                                            horizontal_flip=True,
                                            rotation_range=10,
                                            rescale=1. / 256).flow_from_files(model_config.train_files, mode="fit",
                                                                              target_size=model_config.image_size,
                                                                              batch_size=model_config.train_batch_size,
                                                                              shuffle=True,
                                                                              label_position=model_config.label_position)
    val_flow = data_loader.KerasGenerator(model_config=model_config,
                                          featurewise_center=True,
                                          featurewise_std_normalization=True,
                                          width_shift_range=0.15,
                                          height_shift_range=0.1,
                                          horizontal_flip=True,
                                          rotation_range=10,
                                          rescale=1. / 256).flow_from_files(model_config.val_files, mode="fit",
                                                                            target_size=model_config.image_size,
                                                                            batch_size=model_config.val_batch_size,
                                                                            shuffle=True,
                                                                            label_position=model_config.label_position)

    start = time.time()
    print("####### start train model #######")
    for i in range(len(model_config.epoch)):
        print(
            "lr=%f, freeze layers=%2f epoch=%d" % (
                model_config.lr[i], model_config.freeze_layers[i], model_config.epoch[i]))
        clr = keras_util.CyclicLrCallback(base_lr=model_config.lr[i], max_lr=model_config.lr[i] * 5,
                                          step_size=model_config.get_steps_per_epoch() / 2)

        if i == 0:
            model = get_model()
            model.fit_generator(generator=train_flow,
                                steps_per_epoch=model_config.get_steps_per_epoch(),
                                epochs=model_config.epoch[i],
                                validation_data=val_flow,
                                validation_steps=len(model_config.val_files) / model_config.val_batch_size,
                                workers=16,
                                verbose=1,
                                callbacks=[tensorboard, checkpoint, clr])
        else:
            model = get_model()
            model.load_weights(model_config.tem_model_file)
            model.fit_generator(generator=train_flow,
                                steps_per_epoch=model_config.get_steps_per_epoch(),
                                epochs=model_config.epoch[i],
                                initial_epoch=model_config.epoch[i - 1],
                                validation_data=val_flow,
                                validation_steps=len(model_config.val_files) / model_config.val_batch_size,
                                workers=16,
                                verbose=1,
                                callbacks=[tensorboard, checkpoint, clr])

        model.save_weights(model_config.tem_model_file)
        del model

    print("####### train model spend %d seconds #######" % (time.time() - start))

    model = get_model()
    for i in range(1, model_config.epoch[-1] + 1):
        model.load_weights(model_config.get_weights_path(i))
        keras_util.evaluate(model, model_config.val_files, y_valid, model_config.get_weights_path(i), model_config)
