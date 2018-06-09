import os
import time

import keras
import numpy as np
from keras.layers import Dense, BatchNormalization

import config
from util import data_loader
from util import metrics
from util import path

RESOLUTION = 224
FIRST_EPOCH = 2
SECOND_EPOCH = 5
THIRD_EPOCH = 10
FOURTH_EPOCH = 10

TRAIN_BATCH_SIZE = 32
VAL_BATCH_SIZE = 128
PREDICT_BATCH_SIZE = 128

K_FOLD_FILE = "1.txt"
VAL_INDEX = 1
BASE_DIR = "./record/val1/"
MODEL_FILE = BASE_DIR + 'weights.hdf5'
SAVE_MODEL_FORMAT = BASE_DIR + "weights.{epoch:03d}-{val_smooth_f2_score:.4f}.hdf5"

IMAGE_MEAN_FILE = path.get_image_mean_file(K_FOLD_FILE, VAL_INDEX,
                                           data_type=[config.DATA_TYPE_ORIGINAL])
IMAGE_STD_FILE = path.get_image_std_file(K_FOLD_FILE, VAL_INDEX,
                                         data_type=[config.DATA_TYPE_ORIGINAL])

train_files, val_files = data_loader.get_k_fold_files(K_FOLD_FILE, VAL_INDEX,
                                                      [config.DATA_TYPE_ORIGINAL])

# val_files = val_files[:128]

y_train = np.array(data_loader.get_labels(train_files), np.bool)
y_valid = np.array(data_loader.get_labels(val_files), np.bool)


def get_model(freeze_layers=None, lr=0.001):
    base_model = keras.applications.InceptionResNetV2(weights="imagenet", include_top=True)

    base_model.layers.pop()
    x = Dense(512, activation='relu')(base_model.layers[-1].output)
    x = BatchNormalization()(x)
    predictions = Dense(13, activation='sigmoid')(x)
    model = keras.Model(inputs=base_model.input, outputs=predictions)

    if freeze_layers is None:
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
    print("model contains %d layers" % len(model.layers))
    return model


def check_mean_std_file(datagen: data_loader.KerasGenerator):
    print("check mean file [%s] and std file[%s]" % (IMAGE_MEAN_FILE, IMAGE_STD_FILE))
    if not os.path.exists(IMAGE_STD_FILE) or not os.path.exists(IMAGE_STD_FILE):
        datagen.calc_image_global_mean_std(train_files, 1, RESOLUTION)
        datagen.save_image_global_mean_std(IMAGE_MEAN_FILE, IMAGE_STD_FILE)


def evaluate(model: keras.Model, pre_files, y):
    from sklearn.metrics import fbeta_score

    pre_datagen = data_loader.KerasGenerator(featurewise_center=True,
                                             featurewise_std_normalization=True,
                                             )
    check_mean_std_file(pre_datagen)
    pre_datagen.load_image_global_mean_std(IMAGE_MEAN_FILE, IMAGE_STD_FILE)
    pre_flow = pre_datagen.flow_from_files(pre_files, mode="predict",
                                           target_size=(RESOLUTION, RESOLUTION),
                                           batch_size=PREDICT_BATCH_SIZE)

    start = time.time()
    y_pred = model.predict_generator(pre_flow, steps=len(pre_files) / PREDICT_BATCH_SIZE, verbose=1)
    print("####### predict %d images spend %d seconds ######"
          % (len(pre_files), time.time() - start))
    start = time.time()
    greedy_score, greedy_threshold = metrics.greedy_f2_score(y, y_pred)
    print("####### search greedy threshold spend %d seconds ######"
          % (time.time() - start))
    start = time.time()
    bs_score, bs_threshold = metrics.best_f2_score(y, y_pred)
    print("####### search basonshopping threshold spend %d seconds ######"
          % (time.time() - start))
    print("####### Smooth F2-Score is %f #######"
          % metrics.smooth_f2_score_np(y, y_pred))
    print("####### F2-Score with threshold 0.2 is %f #######"
          % fbeta_score(y, (np.array(y_pred) > 0.2).astype(np.int8), beta=2, average='samples'))
    print("####### F2-Score with threshold 0.1 is %f #######"
          % fbeta_score(y, (np.array(y_pred) > 0.1).astype(np.int8), beta=2, average='samples'))
    print("####### Basonhopping F2-Score is %f #######" % bs_score)
    print("####### Greedy F2-Score is %f #######" % greedy_score)


    if bs_score > greedy_score:
        y_pred_best = (np.array(y_pred) > bs_threshold).astype(np.int8)
    else:
        y_pred_best = (np.array(y_pred) > greedy_threshold).astype(np.int8)
    with open(BASE_DIR + "evaluate.txt", "w+") as f:
        f.write("basonshopping threshold: ")
        f.write(",".join([str(j) for j in bs_threshold]))
        f.write("\n")
        f.write("greedy threshold: ")
        f.write(",".join([str(j) for j in greedy_threshold]))
        f.write("\n")
        for i in range(len(pre_files)):
            f.write(pre_files[i])
            f.write(",")
            f.write(",".join([str(j) for j in list(y_pred_best[i])]))
            f.write("\n")


# tensorboard = keras.callbacks.TensorBoard(log_dir=BASE_DIR)
# checkpoint = keras.callbacks.ModelCheckpoint(filepath=SAVE_MODEL_FORMAT,
#                                              monitor="val_smooth_f2_score",
#                                              save_weights_only=True)
#
# train_datagen = data_loader.KerasGenerator(featurewise_center=True,
#                                            featurewise_std_normalization=True,
#                                            width_shift_range=0.15,
#                                            horizontal_flip=True,
#                                            rotation_range=15,
#                                            )
# val_datagen = data_loader.KerasGenerator(featurewise_center=True,
#                                          featurewise_std_normalization=True,
#                                          width_shift_range=0.15,
#                                          horizontal_flip=True,
#                                          rotation_range=15,
#                                          )
#
# check_mean_std_file(train_datagen)
# train_datagen.load_image_global_mean_std(IMAGE_MEAN_FILE, IMAGE_STD_FILE)
# val_datagen.load_image_global_mean_std(IMAGE_MEAN_FILE, IMAGE_STD_FILE)
#
# train_flow = train_datagen.flow_from_files(train_files, mode="fit",
#                                            target_size=(RESOLUTION, RESOLUTION),
#                                            batch_size=TRAIN_BATCH_SIZE)
# val_flow = val_datagen.flow_from_files(val_files, mode="fit",
#                                        target_size=(RESOLUTION, RESOLUTION),
#                                        batch_size=VAL_BATCH_SIZE)
#
# model = get_model(lr=0.001)
# start = time.time()
# print("####### start train model #######")
# model.fit_generator(generator=train_flow,
#                     steps_per_epoch=len(train_files) / TRAIN_BATCH_SIZE,
#                     epochs=2,
#                     validation_data=val_flow,
#                     validation_steps=len(val_files) / VAL_BATCH_SIZE,
#                     workers=16,
#                     verbose=1,
#                     callbacks=[tensorboard, checkpoint])
#
# print("####### train model spend %d seconds #######" % (time.time() - start))
# model.save_weights(MODEL_FILE)
# del model
#
# model = get_model(freeze_layers=600, lr=0.0001)
# model.load_weights(MODEL_FILE)
# start = time.time()
# print("####### start train model #######")
# model.fit_generator(generator=train_flow,
#                     steps_per_epoch=len(train_files) / TRAIN_BATCH_SIZE,
#                     epochs=7,
#                     initial_epoch=2,
#                     validation_data=val_flow,
#                     validation_steps=len(val_files) / VAL_BATCH_SIZE,
#                     workers=16,
#                     verbose=1,
#                     callbacks=[tensorboard, checkpoint])
# print("####### train model spend %d seconds #######" % (time.time() - start))
# evaluate(model, val_files, y_valid)
# model.save_weights(MODEL_FILE)
# del model
#
# model = get_model(freeze_layers=400, lr=0.00001)
# model.load_weights(MODEL_FILE)
# start = time.time()
# print("####### start train model #######")
# model.fit_generator(generator=train_flow,
#                     steps_per_epoch=len(train_files) / TRAIN_BATCH_SIZE,
#                     epochs=12,
#                     initial_epoch=7,
#                     validation_data=val_flow,
#                     validation_steps=len(val_files) / VAL_BATCH_SIZE,
#                     workers=16,
#                     verbose=1,
#                     callbacks=[tensorboard, checkpoint])
# print("####### train model spend %d seconds #######" % (time.time() - start))
# evaluate(model, val_files, y_valid)
# model.save_weights(MODEL_FILE)
# del model
#
# model = get_model(freeze_layers=200, lr=0.00001)
# model.load_weights(MODEL_FILE)
# start = time.time()
# print("####### start train model #######")
# model.fit_generator(generator=train_flow,
#                     steps_per_epoch=len(train_files) / TRAIN_BATCH_SIZE,
#                     epochs=17,
#                     initial_epoch=12,
#                     validation_data=val_flow,
#                     validation_steps=len(val_files) / VAL_BATCH_SIZE,
#                     workers=16,
#                     verbose=1,
#                     callbacks=[tensorboard, checkpoint])
# print("####### train model spend %d seconds #######" % (time.time() - start))
# evaluate(model, val_files, y_valid)
# model.save_weights(MODEL_FILE)
# del model
#
# model = get_model(freeze_layers=100, lr=0.00001)
# model.load_weights(MODEL_FILE)
# start = time.time()
# print("####### start train model #######")
# model.fit_generator(generator=train_flow,
#                     steps_per_epoch=len(train_files) / TRAIN_BATCH_SIZE,
#                     epochs=22,
#                     initial_epoch=17,
#                     validation_data=val_flow,
#                     validation_steps=len(val_files) / VAL_BATCH_SIZE,
#                     workers=16,
#                     verbose=1,
#                     callbacks=[tensorboard, checkpoint])
# print("####### train model spend %d seconds #######" % (time.time() - start))
# evaluate(model, val_files, y_valid)
# model.save_weights(MODEL_FILE)
# del model
#
# model = get_model(freeze_layers=0, lr=0.00001)
# model.load_weights(MODEL_FILE)
# start = time.time()
# print("####### start train model #######")
# model.fit_generator(generator=train_flow,
#                     steps_per_epoch=len(train_files) / TRAIN_BATCH_SIZE,
#                     epochs=27,
#                     initial_epoch=22,
#                     validation_data=val_flow,
#                     validation_steps=len(val_files) / VAL_BATCH_SIZE,
#                     workers=16,
#                     verbose=1,
#                     callbacks=[tensorboard, checkpoint])
# print("####### train model spend %d seconds #######" % (time.time() - start))
# evaluate(model, val_files, y_valid)


model = get_model()
model.load_weights("./record/val1/weights.006-0.7943.hdf5")
evaluate(model, val_files, y_valid)
model.load_weights("./record/val1/weights.007-0.8046.hdf5")
evaluate(model, val_files, y_valid)
model.load_weights("./record/val1/weights.008-0.8113.hdf5")
evaluate(model, val_files, y_valid)
model.load_weights("./record/val1/weights.009-0.8141.hdf5")
evaluate(model, val_files, y_valid)
model.load_weights("./record/val1/weights.010-0.8158.hdf5")
evaluate(model, val_files, y_valid)
model.load_weights("./record/val1/weights.011-0.8183.hdf5")
evaluate(model, val_files, y_valid)