"""from model 9"""
import math
import os
import queue
import time

import keras
from keras.layers import Dense

import config
from util import data_loader
from util import keras_util
from util import metrics
from util.keras_util import KerasModelConfig

model_config = KerasModelConfig(k_fold_file="1.txt",
                                model_path=os.path.abspath(__file__),
                                image_resolution=324,
                                data_type=[config.DATA_TYPE_ORIGINAL],
                                label_position=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                                label_color_augment=[0, 1, 3, 5, 6, 7, 9, 10, 11, 12],
                                label_up_sampling=[20, 0, 0, 10, 0, 0, 10, 0, 0, 0, 0, 0, 10],
                                downsampling=0.8,
                                train_batch_size=[16, 16],
                                val_batch_size=128,
                                predict_batch_size=128,
                                epoch=[2, 10],
                                lr=[0.0001, 0.00001],
                                freeze_layers=[5, 5],
                                tta_flip=True,
                                data_visualization=True)


def get_model(freeze_layers=-1, lr=0.01, output_dim=1, weights="imagenet"):
    base_model = keras.applications.Xception(include_top=False, weights=weights,
                                             input_shape=model_config.image_shape, pooling="avg")

    x = base_model.output
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
                  optimizer=keras.optimizers.Adam(lr=lr),
                  metrics=['accuracy', metrics.smooth_f2_score, metrics.smooth_f2_score_02_macro,
                           metrics.smooth_f2_score_02])
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
    model_config.save_log("####### start train model")

    init_stage = model_config.get_init_stage()
    model_config.save_log("####### init stage is %d" % init_stage)

    for i in range(init_stage, len(model_config.epoch)):
        model_config.save_log("####### lr=%f, freeze layers=%2f epoch=%d" % (
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
                                                rescale=1. / 256).flow_from_files(model_config.train_files, mode="fit",
                                                                                  target_size=model_config.image_size,
                                                                                  batch_size=
                                                                                  model_config.train_batch_size[i],
                                                                                  shuffle=True,
                                                                                  label_position=model_config.label_position)

        if i == 0:
            model_config.save_log("####### initial epoch is 0, end epoch is %d" % model_config.epoch[i])
            model = get_model(freeze_layers=model_config.freeze_layers[i], lr=model_config.lr[i],
                              output_dim=len(model_config.label_position))
            model.fit_generator(generator=train_flow,
                                steps_per_epoch=model_config.get_steps_per_epoch(i),
                                epochs=model_config.epoch[i],
                                workers=16,
                                verbose=1,
                                callbacks=[checkpoint, clr, tensorboard])
        else:
            model = get_model(freeze_layers=model_config.freeze_layers[i], output_dim=len(model_config.label_position),
                              lr=model_config.lr[i], weights=None)

            if i == init_stage:
                model_config.save_log(
                    "####### load weight file: %s" % model_config.get_weights_path(model_config.initial_epoch))
                model.load_weights(model_config.get_weights_path(model_config.initial_epoch))

                model_config.save_log("####### initial epoch is %d, end epoch is %d" % (
                    model_config.initial_epoch, model_config.epoch[i]))
                model.fit_generator(generator=train_flow,
                                    steps_per_epoch=model_config.get_steps_per_epoch(i),
                                    epochs=model_config.epoch[i],
                                    initial_epoch=model_config.initial_epoch,
                                    workers=16,
                                    verbose=1,
                                    callbacks=[checkpoint, clr, tensorboard])
            else:
                model_config.save_log(
                    "####### load weight file: %s" % model_config.get_weights_path(model_config.epoch[i - 1]))
                model.load_weights(model_config.get_weights_path(model_config.epoch[i - 1]))

                model_config.save_log(
                    "####### initial epoch is %d, end epoch is %d" % (model_config.epoch[i - 1], model_config.epoch[i]))
                model.fit_generator(generator=train_flow,
                                    steps_per_epoch=model_config.get_steps_per_epoch(i),
                                    epochs=model_config.epoch[i],
                                    initial_epoch=model_config.epoch[i - 1],
                                    workers=16,
                                    verbose=1,
                                    callbacks=[checkpoint, clr, tensorboard])

    model_config.save_log("####### train model spend %d seconds" % (time.time() - start))
    model_config.save_log(
        "####### train model spend %d seconds average" % ((time.time() - start) / model_config.epoch[-1]))
