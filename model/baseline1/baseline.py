
import os
import time

import keras
import numpy as np
from keras import Sequential
from keras.layers import Dense, GlobalAveragePooling2D, Conv2D, BatchNormalization, MaxPooling2D, Flatten
from tqdm import tqdm

import config
from util import data_loader
from util import metrics
from util import path

RESOLUTION = 224
THRESHOLD = 0.2
EPOCH = 20
TRAIN_BATCH_SIZE = 32
VAL_BATCH_SIZE = 128
PREDICT_BATCH_SIZE = 128

K_FOLD_FILE = "1.txt"
VAL_INDEX = 1
BASE_DIR = "./record/5/"
MODEL_FILE = BASE_DIR + 'weights.018-0.7310.hdf5'
SAVE_MODEL_FORMAT = BASE_DIR + "weights.{epoch:03d}-{val_smooth_f2_score:.4f}.hdf5"

IMAGE_MEAN_FILE = path.get_image_mean_file(K_FOLD_FILE, VAL_INDEX, RESOLUTION)
IMAGE_STD_FILE = path.get_image_std_file(K_FOLD_FILE, VAL_INDEX, RESOLUTION)

train_files, val_files = data_loader.get_k_fold_files(K_FOLD_FILE, VAL_INDEX,
                                                      [config.DATA_TYPE_ORIGINAL, config.DATA_TYPE_SEGMENTED])

# # 取少量数据看模型是否run的起来
train_files = train_files[:128]
# val_files = val_files[:64]


y_train = np.array(data_loader.get_labels(train_files), np.bool)
y_valid = np.array(data_loader.get_labels(val_files), np.bool)
def get_model():
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
    model.add(Dense(13, activation='sigmoid'))

    model.compile(loss="binary_crossentropy",
                  optimizer=keras.optimizers.SGD(lr=0.0001, momentum=0.9),
                  metrics=['accuracy', metrics.smooth_f2_score])

    # 通过模块文件，读取完整的模型
    # if os.path.isfile(MODEL_FILE):
    #     print('####### Loading model from cache #######')
    #     model = load_model(MODEL_FILE, custom_objects={'smooth_f2_score': metrics.smooth_f2_score,
    #                                                    'logloss_and_f2score': metrics.logloss_and_f2score})

    return model

def get_resnet():
    base_model = keras.applications.ResNet50(weights="imagenet", include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(13, activation='sigmoid')(x)
    model = keras.Model(inputs=base_model.input, outputs=predictions)
    model.compile(loss="binary_crossentropy",
                  optimizer=keras.optimizers.SGD(lr=0.0001, momentum=0.9),
                  metrics=['accuracy', metrics.smooth_f2_score])
    return model


def train(model):
    tensorboard = keras.callbacks.TensorBoard(log_dir=BASE_DIR)
    checkpoint = keras.callbacks.ModelCheckpoint(filepath=SAVE_MODEL_FORMAT,
                                                 monitor="val_smooth_f2_score",
                                                 save_weights_only=True)
    train_datagen = data_loader.KerasGenerator()
    val_datagen = data_loader.KerasGenerator()
    # train_datagen = data_loader.KerasGenerator(featurewise_center=True,
    #                                            featurewise_std_normalization=True,
    #                                            width_shift_range=0.15,
    #                                            horizontal_flip=True,
    #                                            rotation_range=15,
    #                                            rescale=config.IMAGE_RESCALE
    #                                            )
    # val_datagen = data_loader.KerasGenerator(featurewise_center=True,
    #                                          featurewise_std_normalization=True,
    #                                          width_shift_range=0.15,
    #                                          horizontal_flip=True,
    #                                          rotation_range=15,
    #                                          rescale=config.IMAGE_RESCALE
    #                                          )

    # check_mean_std_file(train_datagen)
    # train_datagen.load_image_mean_std(IMAGE_MEAN_FILE, IMAGE_STD_FILE)
    # val_datagen.load_image_mean_std(IMAGE_MEAN_FILE, IMAGE_STD_FILE)

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


def check_mean_std_file(datagen: data_loader.KerasGenerator):
    if not os.path.exists(IMAGE_STD_FILE) or not os.path.exists(IMAGE_STD_FILE):
        datagen.calc_image_mean_std(train_files, config.IMAGE_RESCALE, RESOLUTION)
        datagen.save_image_mean_std(IMAGE_MEAN_FILE, IMAGE_STD_FILE)


def evaluate(model: keras.Model, pre_files, y):
    from sklearn.metrics import fbeta_score

    pre_datagen = data_loader.KerasGenerator()
    # pre_datagen = data_loader.KerasGenerator(featurewise_center=True,
    #                                          featurewise_std_normalization=True,
    #                                          rescale=1. / 255
    #                                          )
    # check_mean_std_file(pre_datagen)
    # pre_datagen.load_image_mean_std(IMAGE_MEAN_FILE, IMAGE_STD_FILE)

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
          % fbeta_score(y, (np.array(y_pred) > THRESHOLD).astype(np.int8), beta=2, average='samples'))
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
            f2_2 = fbeta_score(y, np.array(y_pred) > THRESHOLD, beta=2, average='samples')
            f2_basinhopping, threshold = metrics.best_f2_score(y, y_pred)
            with open(path + "predict_all.txt", "a") as f:
                f.write(
                    "[%s]\t Sample=%d F2_smooth=%.4f,  F2_0.2=%.4f,  F2_basinhopping=%.4f\n" % (
                        weight_file, sample_number, f2_smooth, f2_2, f2_basinhopping))

model = get_model()
# model = get_resnet()
model.load_weights(MODEL_FILE)
# train(model)
evaluate(model, val_files, y_valid)
# evaluate(model, train_files, x_train, y_train)
