import os
import time

import keras
import numpy as np
from keras import Sequential
from keras.layers import Dense, Conv2D, BatchNormalization, MaxPooling2D, Flatten

import config
from util import data_loader
from util import metrics
from util import path

RESOLUTION = 224
TRAIN_BATCH_SIZE = 32
VAL_BATCH_SIZE = 64
PREDICT_BATCH_SIZE = 128
EPOCH = 60

LABEL_POSITION = [3, 4]

K_FOLD_FILE = "1.txt"
VAL_INDEX = 1
BASE_DIR = "./record/val1/"
MODEL_FILE = BASE_DIR + 'weights.hdf5'
SAVE_MODEL_FORMAT = BASE_DIR + "%sweights.{epoch:03d}.hdf5" % str([str(i) for i in LABEL_POSITION])

# 此处写224是为了获取在224上预计算的均值和std（global mean、std与图像尺寸无关，还未修改相关代码）
IMAGE_MEAN_FILE = path.get_image_mean_file(K_FOLD_FILE, VAL_INDEX,
                                           data_type=[config.DATA_TYPE_SEGMENTED])
IMAGE_STD_FILE = path.get_image_std_file(K_FOLD_FILE, VAL_INDEX,
                                         data_type=[config.DATA_TYPE_SEGMENTED])

train_files, val_files = data_loader.get_k_fold_files(K_FOLD_FILE, VAL_INDEX,
                                                      [config.DATA_TYPE_SEGMENTED])


# # 取部分样本，验证流程正确性
# train_files = train_files[:64]
# val_files = val_files[:64]

def get_model(output_dim=1):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(RESOLUTION, RESOLUTION, 3)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(output_dim, activation='sigmoid'))

    model.compile(loss="binary_crossentropy",
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy', metrics.smooth_f2_score])
    model.summary()
    return model


def evaluate(model: keras.Model, pre_files, y, weight_name):
    from sklearn.metrics import fbeta_score

    pre_datagen = data_loader.KerasGenerator(featurewise_center=True,
                                             featurewise_std_normalization=True,
                                             rescale=1. / 256)

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
    greedy_score, greedy_threshold = metrics.greedy_f2_score(y, y_pred, label_num=len(LABEL_POSITION))
    print("####### search greedy threshold spend %d seconds ######"
          % (time.time() - start))
    print("####### Smooth F2-Score is %f #######"
          % metrics.smooth_f2_score_np(y, y_pred))
    if len(LABEL_POSITION) > 1:
        print("####### F2-Score with threshold 0.2 is %f #######"
              % fbeta_score(y, (np.array(y_pred) > 0.2).astype(np.int8), beta=2, average='samples'))
        print("####### F2-Score with threshold 0.1 is %f #######"
              % fbeta_score(y, (np.array(y_pred) > 0.1).astype(np.int8), beta=2, average='samples'))
    else:
        print("####### F2-Score with threshold 0.2 is %f #######"
              % fbeta_score(y, (np.array(y_pred) > 0.2).astype(np.int8), beta=2))
        print("####### F2-Score with threshold 0.1 is %f #######"
              % fbeta_score(y, (np.array(y_pred) > 0.1).astype(np.int8), beta=2))

    print("####### Greedy F2-Score is %f #######" % greedy_score)

    with open(BASE_DIR + "evaluate%s.txt" % str([str(i) for i in LABEL_POSITION]), "a") as f:
        f.write("\n\n")
        f.write("weight: %s" % weight_name)
        f.write("Smooth F2-Score: %f\n"
                % metrics.smooth_f2_score_np(y, y_pred))

        if len(LABEL_POSITION) > 1:
            f.write("F2-Score with threshold 0.2: %f\n"
                    % fbeta_score(y, (np.array(y_pred) > 0.2).astype(np.int8), beta=2, average='samples'))
            print("F2-Score with threshold 0.1: %f\n"
                  % fbeta_score(y, (np.array(y_pred) > 0.1).astype(np.int8), beta=2, average='samples'))
        else:
            f.write("F2-Score with threshold 0.2: %f\n"
                    % fbeta_score(y, (np.array(y_pred) > 0.2).astype(np.int8), beta=2))
            print("F2-Score with threshold 0.1: %f\n"
                  % fbeta_score(y, (np.array(y_pred) > 0.1).astype(np.int8), beta=2))

        f.write("Greedy F2-Score is: %f\n" % greedy_score)
        f.write("Greedy threshold: ")
        f.write(",".join([str(j) for j in greedy_threshold]))
        f.write("\n")

        greedy_threshold_all = []
        for i in range(y_pred.shape[-1]):
            smooth_f2 = metrics.smooth_f2_score_np(y[:, i], y_pred[:, i])
            greedy_f2, greedy_threshold = metrics.greedy_f2_score(y[:, i], y_pred[:, i], 1)
            bason_f2, bason_threshold = metrics.best_f2_score(y[:, i], y_pred[:, i], 1)
            print("[label %d]\tsmooth-f2=%4f   BFGS-f2=%4f[%4f]   greedy-f2=%4f[%4f]" % (
                LABEL_POSITION[i], smooth_f2, bason_f2, bason_threshold[0], greedy_f2, greedy_threshold[0]))
            f.write("[label %d]\tsmooth-f2=%4f   BFGS-f2=%4f[%4f]   greedy-f2=%4f[%4f]\n" % (
                LABEL_POSITION[i], smooth_f2, bason_f2, bason_threshold[0], greedy_f2, greedy_threshold[0]))
            greedy_threshold_all.append(greedy_threshold)


def check_mean_std_file(datagen: data_loader.KerasGenerator):
    if not os.path.exists(IMAGE_STD_FILE) or not os.path.exists(IMAGE_STD_FILE):
        datagen.calc_image_global_mean_std(train_files)
        datagen.save_image_global_mean_std(IMAGE_MEAN_FILE, IMAGE_STD_FILE)


# for i in range(13):
LABEL_POSITION = [4]

y_train = np.array(data_loader.get_labels(train_files), np.bool)[:, LABEL_POSITION]
y_valid = np.array(data_loader.get_labels(val_files), np.bool)[:, LABEL_POSITION]

SAVE_MODEL_FORMAT = BASE_DIR + "%sweights.{epoch:03d}.hdf5" % str([str(i) for i in LABEL_POSITION])
tensorboard = keras.callbacks.TensorBoard(log_dir=BASE_DIR)
checkpoint = keras.callbacks.ModelCheckpoint(filepath=SAVE_MODEL_FORMAT,
                                             save_weights_only=True)

train_datagen = data_loader.KerasGenerator(featurewise_center=True,
                                           featurewise_std_normalization=True,
                                           width_shift_range=0.15,
                                           height_shift_range=0.1,
                                           horizontal_flip=True,
                                           rotation_range=10,
                                           rescale=1. / 256)
val_datagen = data_loader.KerasGenerator(featurewise_center=True,
                                         featurewise_std_normalization=True,
                                         width_shift_range=0.15,
                                         height_shift_range=0.1,
                                         horizontal_flip=True,
                                         rotation_range=10,
                                         rescale=1. / 256)

check_mean_std_file(train_datagen)
train_datagen.load_image_global_mean_std(IMAGE_MEAN_FILE, IMAGE_STD_FILE)
val_datagen.load_image_global_mean_std(IMAGE_MEAN_FILE, IMAGE_STD_FILE)

train_flow = train_datagen.flow_from_files(train_files, mode="fit",
                                           target_size=(RESOLUTION, RESOLUTION),
                                           batch_size=TRAIN_BATCH_SIZE,
                                           shuffle=True,
                                           label_position=LABEL_POSITION)
val_flow = val_datagen.flow_from_files(val_files, mode="fit",
                                       target_size=(RESOLUTION, RESOLUTION),
                                       batch_size=VAL_BATCH_SIZE,
                                       shuffle=True,
                                       label_position=LABEL_POSITION)

model = get_model(len(LABEL_POSITION))
start = time.time()

print("####### start train model #######")
model.fit_generator(generator=train_flow,
                    steps_per_epoch=len(train_files) / TRAIN_BATCH_SIZE,
                    epochs=EPOCH,
                    validation_data=val_flow,
                    validation_steps=len(val_files) / VAL_BATCH_SIZE,
                    workers=12,
                    verbose=1,
                    callbacks=[tensorboard, checkpoint])

print("####### train model spend %d seconds #######" % (time.time() - start))

model = get_model(len(LABEL_POSITION))
for i in range(1, EPOCH + 1):
    if i < 10:
        model.load_weights("./record/val1/%sweights.00%d.hdf5" % (str([str(j) for j in LABEL_POSITION]), i))
        evaluate(model, val_files, y_valid, "%sweights.00%d.hdf5" % (str([str(j) for j in LABEL_POSITION]), i))
    else:
        model.load_weights("./record/val1/%sweights.0%d.hdf5" % (str([str(j) for j in LABEL_POSITION]), i))
        evaluate(model, val_files, y_valid, "%sweights.0%d.hdf5" % (str([str(j) for j in LABEL_POSITION]), i))
