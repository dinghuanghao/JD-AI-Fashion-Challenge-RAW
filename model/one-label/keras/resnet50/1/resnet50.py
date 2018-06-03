import os
import time

import keras
import numpy as np
from keras.layers import Dense, BatchNormalization, Activation

import config
from util import data_loader
from util import keras_util
from util import metrics
from util.keras_util import KerasModelConfig

model_config = KerasModelConfig(k_fold_file="1.txt",
                                val_index=1,
                                image_resolution=224,
                                data_type=[config.DATA_TYPE_SEGMENTED],
                                model_dir=os.path.dirname(os.path.abspath(__file__)),
                                record_sub_dir="val1",
                                label_position=[4],
                                train_batch_size=32,
                                val_batch_size=128,
                                predict_batch_size=128,
                                epoch=[1, 10, 15],
                                lr=[0.001, 0.0001, 0.00001],
                                freeze_layers=[-1, 20, 0])

# 取部分样本，验证流程正确性
# model_config.train_files = model_config.train_files[:64]
# model_config.val_files = model_config.val_files[:64]


def get_model(freeze_layers=-1, lr=0.01, output_dim=1):
    base_model = keras.applications.ResNet50(weights="imagenet", include_top=False,
                                             input_shape=model_config.image_shape, pooling="avg")
    x = base_model.output
    x = Dense(256, activation="relu", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    predictions = Dense(output_dim, activation='sigmoid')(x)
    model = keras.Model(inputs=base_model.input, outputs=predictions)

    if freeze_layers is -1:
        print("freeze all basic layers, lr=%f" % lr)

        for layer in base_model.layers:
            layer.trainable = False
    else:
        for layer in range(freeze_layers):
            base_model.layers[layer].train_layer = False
        print("freeze %d basic layers, lr=%f" % (freeze_layers, lr))

    model.compile(loss="binary_crossentropy",
                  optimizer=keras.optimizers.Adam(lr),
                  metrics=['accuracy', metrics.smooth_f2_score])
    print("model have %d layers" % len(model.layers))
    return model


y_train = np.array(data_loader.get_labels(model_config.train_files), np.bool)[:, model_config.label_position]
y_valid = np.array(data_loader.get_labels(model_config.val_files), np.bool)[:, model_config.label_position]

tensorboard = keras.callbacks.TensorBoard(log_dir=model_config.record_dir)
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
    print("lr=%f, freeze layers=%d epoch=%d" %(model_config.lr[i], model_config.freeze_layers[i], model_config.epoch[i]))
    if i == 0:
        model = get_model(lr=model_config.lr[i], output_dim=len(model_config.label_position))
        model.fit_generator(generator=train_flow,
                            steps_per_epoch=len(model_config.train_files) / model_config.train_batch_size,
                            epochs=model_config.epoch[i],
                            validation_data=val_flow,
                            validation_steps=len(model_config.val_files) / model_config.val_batch_size,
                            workers=12,
                            verbose=0,
                            callbacks=[tensorboard, checkpoint])
    else:
        model = get_model(freeze_layers=model_config.freeze_layers[i], output_dim=len(model_config.label_position),
                          lr=model_config.lr[i])
        model.load_weights(model_config.tem_model_file)
        model.fit_generator(generator=train_flow,
                            steps_per_epoch=len(model_config.train_files) / model_config.train_batch_size,
                            epochs=model_config.epoch[i],
                            initial_epoch=model_config.epoch[i - 1],
                            validation_data=val_flow,
                            validation_steps=len(model_config.val_files) / model_config.val_batch_size,
                            workers=12,
                            verbose=0,
                            callbacks=[tensorboard, checkpoint])

    model.save_weights(model_config.tem_model_file)
    del model

print("####### train model spend %d seconds #######" % (time.time() - start))

model = get_model(len(model_config.label_position))
for i in range(1, model_config.epoch[-1] + 1):
    model.load_weights(model_config.get_weights_path(i))
    keras_util.evaluate(model, model_config.val_files, y_valid, model_config.get_weights_path(i), model_config)
