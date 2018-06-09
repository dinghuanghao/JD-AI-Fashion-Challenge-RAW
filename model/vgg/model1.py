# -*- coding:utf-8 -*-  
"""
本模型的目的是尽可能地降低bias：
1. 减小数据量，仅仅用Segmented数据，且不做随机样本生成
2. 用BCE函数，加快收敛
3. 使用BatchNorm加快收敛
4. 第一阶段仅仅训练新增的后两层
"""
import os
import time

import keras
import numpy as np
from keras.layers import Dense, BatchNormalization, Flatten
from tqdm import tqdm

import config
from util import data_loader
from util import metrics
from util import path

RESOLUTION = 224
EPOCH = 15
TRAIN_BATCH_SIZE = 32
VAL_BATCH_SIZE = 128
PREDICT_BATCH_SIZE = 128

K_FOLD_FILE = "1.txt"
VAL_INDEX = 1
BASE_DIR = "./record/model_1/val1"
MODEL_FILE = BASE_DIR + 'weights.018-0.7310.hdf5'
SAVE_MODEL_FORMAT = BASE_DIR + "weights.{epoch:03d}-{val_smooth_f2_score:.4f}.hdf5"

IMAGE_MEAN_FILE = path.get_image_mean_file(K_FOLD_FILE, VAL_INDEX)
IMAGE_STD_FILE = path.get_image_std_file(K_FOLD_FILE, VAL_INDEX)

train_files, val_files = data_loader.get_k_fold_files(K_FOLD_FILE, VAL_INDEX,
                                                      [config.DATA_TYPE_ORIGINAL])

# # 取少量数据看模型是否run的起来
# train_files = train_files[:128]
# val_files = val_files[:64]


y_train = np.array(data_loader.get_labels(train_files), np.bool)
y_valid = np.array(data_loader.get_labels(val_files), np.bool)


def get_model():
    base_model = keras.applications.VGG16(weights="imagenet", include_top=True)

    # 去掉最后一个FC（Softmax层），但是保留了Flatten层，没有做GLobalAverage
    base_model.layers.pop()
    x = Dense(512, activation='relu')(base_model.layers[-1].output)
    x = BatchNormalization()(x)
    predictions = Dense(13, activation='sigmoid')(x)
    model = keras.Model(inputs=base_model.input, outputs=predictions)

    # 只训练顶部的两层
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(loss="binary_crossentropy",
                  optimizer='rmsprop',
                  metrics=['accuracy', metrics.smooth_f2_score])
    return model


def train(model):
    tensorboard = keras.callbacks.TensorBoard(log_dir=BASE_DIR)
    checkpoint = keras.callbacks.ModelCheckpoint(filepath=SAVE_MODEL_FORMAT,
                                                 monitor="val_smooth_f2_score",
                                                 save_weights_only=True)

    train_datagen = data_loader.KerasGenerator()
    val_datagen = data_loader.KerasGenerator()

    train_flow = train_datagen.flow_from_files(train_files, mode="fit",
                                               target_size=(RESOLUTION, RESOLUTION),
                                               batch_size=TRAIN_BATCH_SIZE)
    val_flow = val_datagen.flow_from_files(val_files, mode="fit",
                                           target_size=(RESOLUTION, RESOLUTION),
                                           batch_size=VAL_BATCH_SIZE)

    start = time.time()
    print("####### start train model #######")
    model.fit_generator(generator=train_flow,
                        steps_per_epoch=len(train_files) / TRAIN_BATCH_SIZE,
                        epochs=EPOCH,
                        validation_data=val_flow,
                        validation_steps=len(val_files) / VAL_BATCH_SIZE,
                        workers=16,
                        verbose=1,
                        callbacks=[tensorboard, checkpoint])

    print("####### train model spend %d seconds #######" % (time.time() - start))


def evaluate(model: keras.Model, pre_files, y):
    from sklearn.metrics import fbeta_score

    pre_datagen = data_loader.KerasGenerator()
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


def evaluate_all(path, model, x, y):
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    sample_number = len(x)
    for weight_file in tqdm(files):
        if weight_file.split(".")[-1] == "hdf5":
            model.load_weights(BASE_DIR + weight_file)
            from sklearn.metrics import fbeta_score
            y_pred = model.predict(x, batch_size=PREDICT_BATCH_SIZE)
            f2_smooth = metrics.smooth_f2_score_np(y, y_pred)
            f2_2 = fbeta_score(y, np.array(y_pred) > 0.2, beta=2, average='samples')
            f2_basinhopping, threshold = metrics.best_f2_score(y, y_pred)
            with open(path + "predict_all.txt", "a") as f:
                f.write(
                    "[%s]\t Sample=%d F2_smooth=%.4f,  F2_0.2=%.4f,  F2_basinhopping=%.4f\n" % (
                        weight_file, sample_number, f2_smooth, f2_2, f2_basinhopping))


model = get_model()
# model.load_weights(MODEL_FILE)
train(model)
evaluate(model, val_files, y_valid)
