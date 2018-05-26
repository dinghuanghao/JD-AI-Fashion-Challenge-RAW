"""
本模型的目的是尽可能地降低bias：
1. 减小数据量，仅仅用Segmented数据，且不做随机样本生成
2. 用BCE函数，加快收敛
3. 使用BatchNorm加快收敛
4. 第一阶段仅仅训练新增的后两层
5. 第二阶段训练除了前20层的所有层
"""
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
FIRST_EPOCH = 1
SECOND_EPOCH = 10
TRAIN_BATCH_SIZE = 32
VAL_BATCH_SIZE = 256
PREDICT_BATCH_SIZE = 256

K_FOLD_FILE = "1.txt"
VAL_INDEX = 1
BASE_DIR = "./record/val1/"
MODEL_FILE = BASE_DIR + 'weights.001.hdf5'
SAVE_MODEL_FORMAT = BASE_DIR + "weights.{epoch:03d}-{val_smooth_f2_score:.4f}.hdf5"

IMAGE_MEAN_FILE = path.get_image_mean_file(K_FOLD_FILE, VAL_INDEX, RESOLUTION)
IMAGE_STD_FILE = path.get_image_std_file(K_FOLD_FILE, VAL_INDEX, RESOLUTION)

train_files, val_files = data_loader.get_k_fold_files(K_FOLD_FILE, VAL_INDEX,
                                                      [config.DATA_TYPE_SEGMENTED])

# # 取少量数据看模型是否run的起来
# train_files = train_files[:128]
# val_files = val_files[:64]


y_train = np.array(data_loader.get_labels(train_files), np.bool)
y_valid = np.array(data_loader.get_labels(val_files), np.bool)


def get_model(freeze_layers=None, lr=0.001):
    base_model = keras.applications.ResNet50(weights="imagenet", include_top=True)

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
    return model


def check_mean_std_file(datagen: data_loader.KerasGenerator):
    if not os.path.exists(IMAGE_STD_FILE) or not os.path.exists(IMAGE_STD_FILE):
        datagen.calc_image_mean_std(train_files, config.IMAGE_RESCALE, RESOLUTION)
        datagen.save_image_mean_std(IMAGE_MEAN_FILE, IMAGE_STD_FILE)


def evaluate(model: keras.Model, pre_files, y):
    from sklearn.metrics import fbeta_score

    pre_datagen = data_loader.KerasGenerator(featurewise_center=True,
                                             featurewise_std_normalization=True,
                                             rescale=1. / 255
                                             )
    check_mean_std_file(pre_datagen)
    pre_flow = pre_datagen.flow_from_files(pre_files, mode="predict",
                                           target_size=(RESOLUTION, RESOLUTION),
                                           batch_size=PREDICT_BATCH_SIZE)

    start = time.time()
    y_pred = model.predict_generator(pre_flow, steps=len(pre_files) / PREDICT_BATCH_SIZE, verbose=1)
    print("####### predict %d images spend %d seconds ######"
          % (len(pre_files), time.time() - start))

    start = time.time()
    best_score, threshold = metrics.best_f2_score(y, y_pred)
    print("####### search threshold spend %d seconds ######"
          % (time.time() - start))
    print("####### Smooth F2-Score is %f #######"
          % metrics.smooth_f2_score_np(y, y_pred))
    print("####### F2-Score with threshold 0.2 is %f #######"
          % fbeta_score(y, (np.array(y_pred) > 0.2).astype(np.int8), beta=2, average='samples'))
    print("####### F2-Score with threshold 0.1 is %f #######"
          % fbeta_score(y, (np.array(y_pred) > 0.1).astype(np.int8), beta=2, average='samples'))
    print("####### Best F2-Score is %f #######" % best_score)

    y_pred_best = (np.array(y_pred) > threshold).astype(np.int8)
    with open(BASE_DIR + "evaluate.txt", "w+") as f:
        f.write("threshold: ")
        f.write(",".join([str(j) for j in threshold]))
        f.write("\n")
        for i in range(len(pre_files)):
            f.write(pre_files[i])
            f.write(",")
            f.write(",".join([str(j) for j in list(y_pred_best[i])]))
            f.write("\n")


tensorboard = keras.callbacks.TensorBoard(log_dir=BASE_DIR)
checkpoint = keras.callbacks.ModelCheckpoint(filepath=SAVE_MODEL_FORMAT,
                                             monitor="val_smooth_f2_score",
                                             save_weights_only=True)

train_datagen = data_loader.KerasGenerator(featurewise_center=True,
                                           featurewise_std_normalization=True,
                                           width_shift_range=0.15,
                                           horizontal_flip=True,
                                           rotation_range=15,
                                           rescale=config.IMAGE_RESCALE
                                           )
val_datagen = data_loader.KerasGenerator(featurewise_center=True,
                                         featurewise_std_normalization=True,
                                         width_shift_range=0.15,
                                         horizontal_flip=True,
                                         rotation_range=15,
                                         rescale=config.IMAGE_RESCALE
                                         )

train_flow = train_datagen.flow_from_files(train_files, mode="fit",
                                           target_size=(RESOLUTION, RESOLUTION),
                                           batch_size=TRAIN_BATCH_SIZE)
val_flow = val_datagen.flow_from_files(val_files, mode="fit",
                                       target_size=(RESOLUTION, RESOLUTION),
                                       batch_size=VAL_BATCH_SIZE)

model = get_model(lr=0.001)
start = time.time()
print("####### start train model #######")
model.fit_generator(generator=train_flow,
                    steps_per_epoch=len(train_files) / TRAIN_BATCH_SIZE,
                    epochs=FIRST_EPOCH,
                    validation_data=val_flow,
                    validation_steps=len(val_files) / VAL_BATCH_SIZE,
                    workers=16,
                    verbose=1,
                    callbacks=[tensorboard, checkpoint])

print("####### train model spend %d seconds #######" % (time.time() - start))
model.save_weights(MODEL_FILE)
del model

model = get_model(freeze_layers=20, lr=0.0001)
model.load_weights(MODEL_FILE)
print("####### start train model #######")
model.fit_generator(generator=train_flow,
                    steps_per_epoch=len(train_files) / TRAIN_BATCH_SIZE,
                    epochs=FIRST_EPOCH + SECOND_EPOCH,
                    initial_epoch=FIRST_EPOCH,
                    validation_data=val_flow,
                    validation_steps=len(val_files) / VAL_BATCH_SIZE,
                    workers=16,
                    verbose=1,
                    callbacks=[tensorboard, checkpoint])
print("####### train model spend %d seconds #######" % (time.time() - start))

evaluate(model, val_files, y_valid)
