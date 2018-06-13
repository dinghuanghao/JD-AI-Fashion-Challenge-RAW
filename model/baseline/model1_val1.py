import os
import queue
import time

import keras
from keras import Sequential
from keras.layers import Dense, BatchNormalization, Conv2D, MaxPooling2D, Flatten

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
                                train_batch_size=[32, 32, 32],
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
    evaluate_queue = queue.Queue()
    evaluate_task = keras_util.EvaluateTask(evaluate_queue)
    evaluate_task.setDaemon(True)
    evaluate_task.start()
    checkpoint = keras_util.EvaluateCallback(model_config, evaluate_queue)

    start = time.time()
    print("####### start train model")
    for i in range(len(model_config.epoch)):
        print("####### lr=%f, freeze layers=%2f epoch=%d" % (
            model_config.lr[i], model_config.freeze_layers[i], model_config.epoch[i]))
        clr = keras_util.CyclicLrCallback(base_lr=model_config.lr[i], max_lr=model_config.lr[i] * 5,
                                          step_size=model_config.get_steps_per_epoch(i) / 2)

        train_flow = data_loader.KerasGenerator(model_config=model_config,
                                                featurewise_center=True,
                                                featurewise_std_normalization=True,
                                                width_shift_range=0.15,
                                                height_shift_range=0.1,
                                                horizontal_flip=True,
                                                rotation_range=10,
                                                rescale=1. / 256).flow_from_files(model_config.train_files, mode="fit",
                                                                                  target_size=model_config.image_size,
                                                                                  batch_size=
                                                                                  model_config.train_batch_size[i],
                                                                                  shuffle=True,
                                                                                  label_position=model_config.label_position)

        if i == 0:
            model = get_model()
            model.fit_generator(generator=train_flow,
                                steps_per_epoch=model_config.get_steps_per_epoch(i),
                                epochs=model_config.epoch[i],
                                workers=16,
                                callbacks=[checkpoint, clr])
        else:
            model = get_model()
            print("####### load weight file: %s" % model_config.get_weights_path(model_config.epoch[i - 1]))
            model.load_weights(model_config.get_weights_path(model_config.epoch[i - 1]))
            model.fit_generator(generator=train_flow,
                                steps_per_epoch=model_config.get_steps_per_epoch(i),
                                epochs=model_config.epoch[i],
                                initial_epoch=model_config.epoch[i - 1],
                                workers=16,
                                callbacks=[checkpoint, clr])

    print("####### train model spend %d seconds" % (time.time() - start))
    print("####### train model spend %d seconds average" % ((time.time() - start) / model_config.epoch[-1]))
