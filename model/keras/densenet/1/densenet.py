"""
取消掉图片的单通道均值归一化，并测试800*800尺寸下的结果
"""
import os
import time

import keras
import numpy as np
from keras.layers import Dense, BatchNormalization, Activation

import config
from util import clr_callback
from util import data_loader
from util import metrics
from util import path

RESOLUTION = 224
TRAIN_BATCH_SIZE = 32
VAL_BATCH_SIZE = 64
PREDICT_BATCH_SIZE = 128

K_FOLD_FILE = "1.txt"
VAL_INDEX = 1
BASE_DIR = "./record/val1/"
MODEL_FILE = BASE_DIR + 'weights.hdf5'
SAVE_MODEL_FORMAT = BASE_DIR + "weights.{epoch:03d}.hdf5"

# 此处写224是为了获取在224上预计算的均值和std（global mean、std与图像尺寸无关，还未修改相关代码）
IMAGE_MEAN_FILE = path.get_image_mean_file(K_FOLD_FILE, VAL_INDEX, 224, rescale=256,
                                           data_type=[config.DATA_TYPE_SEGMENTED])
IMAGE_STD_FILE = path.get_image_std_file(K_FOLD_FILE, VAL_INDEX, 224, rescale=256,
                                         data_type=[config.DATA_TYPE_SEGMENTED])

train_files, val_files = data_loader.get_k_fold_files(K_FOLD_FILE, VAL_INDEX,
                                                      [config.DATA_TYPE_SEGMENTED])

y_train = np.array(data_loader.get_labels(train_files), np.bool)
y_valid = np.array(data_loader.get_labels(val_files), np.bool)


def get_model(freeze_layers=None, lr=0.01):
    base_model = keras.applications.DenseNet169(weights="imagenet", include_top=False,
                                                input_shape=(RESOLUTION, RESOLUTION, 3), pooling="max")
    x = base_model.output
    x = Dense(256, activation="relu", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
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
    print("model have %d layers" % len(model.layers))
    return model


def evaluate(model: keras.Model, pre_files, y):
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
    greedy_score, greedy_threshold = metrics.greedy_f2_score(y, y_pred)
    print("####### search greedy threshold spend %d seconds ######"
          % (time.time() - start))
    print("####### Smooth F2-Score is %f #######"
          % metrics.smooth_f2_score_np(y, y_pred))
    print("####### F2-Score with threshold 0.2 is %f #######"
          % fbeta_score(y, (np.array(y_pred) > 0.2).astype(np.int8), beta=2, average='samples'))
    print("####### F2-Score with threshold 0.1 is %f #######"
          % fbeta_score(y, (np.array(y_pred) > 0.1).astype(np.int8), beta=2, average='samples'))
    print("####### Greedy F2-Score is %f #######" % greedy_score)

    y_pred_best = (np.array(y_pred) > greedy_threshold).astype(np.int8)
    with open(BASE_DIR + "evaluate.txt", "w+") as f:
        f.write("greedy threshold: ")
        f.write(",".join([str(j) for j in greedy_threshold]))
        f.write("\n")
        for i in range(len(pre_files)):
            f.write(pre_files[i])
            f.write(",")
            f.write(",".join([str(j) for j in list(y_pred_best[i])]))
            f.write("\n")


def check_mean_std_file(datagen: data_loader.KerasGenerator):
    if not os.path.exists(IMAGE_STD_FILE) or not os.path.exists(IMAGE_STD_FILE):
        datagen.calc_image_global_mean_std(train_files)
        datagen.save_image_mean_std(IMAGE_MEAN_FILE, IMAGE_STD_FILE)


tensorboard = keras.callbacks.TensorBoard(log_dir=BASE_DIR)
checkpoint = keras.callbacks.ModelCheckpoint(filepath=SAVE_MODEL_FORMAT,
                                             save_weights_only=True)

train_datagen = data_loader.KerasGenerator(featurewise_center=True,
                                           featurewise_std_normalization=True,
                                           width_shift_range=0.15,
                                           height_shift_range=0.1,
                                           horizontal_flip=True,
                                           rescale=1. / 256)
val_datagen = data_loader.KerasGenerator(featurewise_center=True,
                                         featurewise_std_normalization=True,
                                         width_shift_range=0.15,
                                         height_shift_range=0.1,
                                         horizontal_flip=True,
                                         rescale=1. / 256)

check_mean_std_file(train_datagen)
train_datagen.load_image_global_mean_std(IMAGE_MEAN_FILE, IMAGE_STD_FILE)
val_datagen.load_image_global_mean_std(IMAGE_MEAN_FILE, IMAGE_STD_FILE)

train_flow = train_datagen.flow_from_files(train_files, mode="fit",
                                           target_size=(RESOLUTION, RESOLUTION),
                                           batch_size=TRAIN_BATCH_SIZE,
                                           shuffle=True)
val_flow = val_datagen.flow_from_files(val_files, mode="fit",
                                       target_size=(RESOLUTION, RESOLUTION),
                                       batch_size=VAL_BATCH_SIZE,
                                       shuffle=True)

clr = clr_callback.CyclicLR(base_lr=0.001, max_lr=0.01, step_size=len(train_files) / TRAIN_BATCH_SIZE / 2)
model = get_model(lr=0.001)
start = time.time()
print("####### start train model #######")
model.fit_generator(generator=train_flow,
                    steps_per_epoch=len(train_files) / TRAIN_BATCH_SIZE,
                    epochs=1,
                    validation_data=val_flow,
                    validation_steps=len(val_files) / VAL_BATCH_SIZE,
                    workers=12,
                    verbose=1,
                    callbacks=[tensorboard, checkpoint, clr])

print("####### train model spend %d seconds #######" % (time.time() - start))
model.save_weights(MODEL_FILE)
del model

clr = clr_callback.CyclicLR(base_lr=0.0001, max_lr=0.0005, step_size=len(train_files) / TRAIN_BATCH_SIZE / 2)
model = get_model(freeze_layers=512, lr=0.0001)
model.load_weights(MODEL_FILE)
start = time.time()
print("####### start train model #######")
model.fit_generator(generator=train_flow,
                    steps_per_epoch=len(train_files) / TRAIN_BATCH_SIZE,
                    epochs=10,
                    initial_epoch=1,
                    validation_data=val_flow,
                    validation_steps=len(val_files) / VAL_BATCH_SIZE,
                    workers=12,
                    verbose=1,
                    callbacks=[tensorboard, checkpoint, clr])
print("####### train model spend %d seconds #######" % (time.time() - start))
evaluate(model, val_files, y_valid)
model.save_weights(MODEL_FILE)
del model

clr = clr_callback.CyclicLR(base_lr=0.00001, max_lr=0.00005, step_size=len(train_files) / TRAIN_BATCH_SIZE / 2)
model = get_model(freeze_layers=0, lr=0.00001)
model.load_weights(MODEL_FILE)
start = time.time()
print("####### start train model #######")
model.fit_generator(generator=train_flow,
                    steps_per_epoch=len(train_files) / TRAIN_BATCH_SIZE,
                    epochs=20,
                    initial_epoch=10,
                    validation_data=val_flow,
                    validation_steps=len(val_files) / VAL_BATCH_SIZE,
                    workers=16,
                    verbose=1,
                    callbacks=[tensorboard, checkpoint, clr])
print("####### train model spend %d seconds #######" % (time.time() - start))
evaluate(model, val_files, y_valid)

# 显示learning rate变化曲线和accuracy曲线
# h = clr.history
# lr = h['lr']
# acc = h['acc']
# x = [i for i in range(len(lr))]
# import matplotlib.pyplot as plt
#
# fig = plt.figure()
# plt.plot(x, lr, 'r')
# plt.plot(x, acc, 'g')
# plt.show()

model = get_model()
model.load_weights("./record/val1/weights.008.hdf5")
evaluate(model, val_files, y_valid)
model.load_weights("./record/val1/weights.010.hdf5")
evaluate(model, val_files, y_valid)
model.load_weights("./record/val1/weights.012.hdf5")
evaluate(model, val_files, y_valid)
model.load_weights("./record/val1/weights.016.hdf5")
evaluate(model, val_files, y_valid)
model.load_weights("./record/val1/weights.020.hdf5")
evaluate(model, val_files, y_valid)
model.load_weights("./record/val1/weights.024.hdf5")
evaluate(model, val_files, y_valid)
model.load_weights("./record/val1/weights.028.hdf5")
evaluate(model, val_files, y_valid)
model.load_weights("./record/val1/weights.032.hdf5")
evaluate(model, val_files, y_valid)
model.load_weights("./record/val1/weights.036.hdf5")
evaluate(model, val_files, y_valid)
model.load_weights("./record/val1/weights.040.hdf5")
evaluate(model, val_files, y_valid)
