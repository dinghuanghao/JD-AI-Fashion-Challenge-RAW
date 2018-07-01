"""
以model 3为基础
"""
import math
import os
import queue
import time

import keras
from keras.layers import Dense, BatchNormalization, Activation

import config
from util import data_loader
from util import keras_util
from util.keras_util import KerasModelConfig

model_config = KerasModelConfig(k_fold_file="1.txt",
                                model_path=os.path.abspath(__file__),
                                image_resolution=224,
                                data_type=[config.DATA_TYPE_ORIGINAL],
                                label_position=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                                label_color_augment=[0, 1, 3, 5, 6, 7, 9, 10, 11, 12],
                                train_batch_size=[16, 16, 16],
                                label_up_sampling=[15, 0, 0, 15, 0, 0, 15, 0, 0, 0, 0, 0, 15],
                                data_visualization=True,
                                downsampling=0.5,
                                val_batch_size=256,
                                predict_batch_size=256,
                                epoch=[2, 5, 10],
                                lr=[0.0005, 0.00005, 0.000005],
                                freeze_layers=[-1, 0.6, 5])


def get_model(freeze_layers=-1, lr=0.01, output_dim=1, weights="imagenet"):
    base_model = keras.applications.DenseNet169(include_top=False, weights=weights,
                                                input_shape=model_config.image_shape, pooling="avg")

    x = base_model.output
    x = Dense(256, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    predictions = Dense(units=output_dim, activation='sigmoid')(x)
    model = keras.Model(inputs=base_model.input, outputs=predictions)

    if freeze_layers == -1:
        print("freeze all basic layers, lr=%f" % lr)

        for layer in base_model.layers:
            layer.trainable = False
    else:
        if freeze_layers < 1:
            freeze_layers = math.floor(len(base_model.layers) * freeze_layers)
        for layer in range(freeze_layers):
            base_model.layers[layer].train_layer = False
        print("freeze %d basic layers, lr=%f" % (freeze_layers, lr))

    model.compile(loss="binary_crossentropy",
                  optimizer=keras.optimizers.Adam(lr=lr))
    # model.summary()
    print("basic model have %d layers" % len(base_model.layers))
    return model


def train():
    evaluate_queue = queue.Queue()
    evaluate_task = keras_util.EvaluateTask(evaluate_queue)
    evaluate_task.setDaemon(True)
    evaluate_task.start()
    checkpoint = keras_util.EvaluateCallback(model_config, evaluate_queue)
    tensorboard = keras_util.TensorBoardCallback(log_dir=model_config.record_dir, log_every=20,
                                                 model_config=model_config)

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
                                                real_transform=True,
                                                rescale=1. / 256). \
            flow_from_files(model_config.train_files,
                            mode="fit",
                            target_size=model_config.image_size,
                            batch_size=model_config.train_batch_size[i],
                            shuffle=True,
                            label_position=model_config.label_position)

        if i == 0:
            model = get_model(freeze_layers=model_config.freeze_layers[i], lr=model_config.lr[i],
                              output_dim=len(model_config.label_position))
            model.fit_generator(generator=train_flow,
                                steps_per_epoch=model_config.get_steps_per_epoch(i),
                                epochs=model_config.epoch[i],
                                workers=16,
                                verbose=0,
                                callbacks=[checkpoint, clr, tensorboard])
        else:
            model = get_model(freeze_layers=model_config.freeze_layers[i], output_dim=len(model_config.label_position),
                              lr=model_config.lr[i], weights=None)
            print("####### load weight file: %s" % model_config.get_weights_path(model_config.epoch[i - 1]))
            model.load_weights(model_config.get_weights_path(model_config.epoch[i - 1]))
            model.fit_generator(generator=train_flow,
                                steps_per_epoch=model_config.get_steps_per_epoch(i),
                                epochs=model_config.epoch[i],
                                initial_epoch=model_config.epoch[i - 1],
                                workers=16,
                                verbose=0,
                                callbacks=[checkpoint, clr, tensorboard])

    print("####### train model spend %d seconds" % (time.time() - start))
    print("####### train model spend %d seconds average" % ((time.time() - start) / model_config.epoch[-1]))
