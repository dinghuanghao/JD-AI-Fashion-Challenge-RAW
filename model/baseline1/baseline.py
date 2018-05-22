import os
import time

import cv2
import keras
import numpy as np
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten, BatchNormalization
from keras.models import Sequential
from tqdm import tqdm

import config
from util import data_loader
from util import metrics

RESOLUTION = 128
THRESHOLD = 0.2
EPOCH = 200
TRAIN_BATCH_SIZE = 32
PREDICT_BATCH_SIZE = 256

BASE_DIR = "./record/3/"
MODEL_FILE = BASE_DIR + 'weights.40-0.73.hdf5'
SAVE_MODEL_FORMAT = BASE_DIR + "weights.{epoch:03d}-{val_smooth_f2_score:.4f}.hdf5"

train_files, val_files = data_loader.get_k_fold_files("baseline.txt", 1, [config.DATA_TYPE_ORIGINAL])

# # 取少量数据看模型是否run的起来
# train_files = train_files[:64]
# val_files = val_files[:64]

x_train = []
x_valid = []
for file in tqdm(train_files, miniters=64):
    img = cv2.imread(file)
    x_train.append(cv2.resize(img, (RESOLUTION, RESOLUTION)))

for file in tqdm(val_files, miniters=64):
    img = cv2.imread(file)
    x_valid.append(cv2.resize(img, (RESOLUTION, RESOLUTION)))

x_train = np.array(x_train, np.float16) / 255.
x_valid = np.array(x_valid, np.float16) / 225.

y_train = data_loader.get_labels(train_files)
y_valid = data_loader.get_labels(val_files)

y_train = np.array(y_train, np.bool)
y_valid = np.array(y_valid, np.bool)

print(x_train.shape, y_train.shape)
print(x_valid.shape, y_valid.shape)


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

    model.compile(loss=metrics.f2_score_loss,
                  optimizer=keras.optimizers.SGD(lr=0.0001, momentum=0.9),
                  metrics=['accuracy', metrics.smooth_f2_score])

    # 通过模块文件，读取完整的模型
    # if os.path.isfile(MODEL_FILE):
    #     print('####### Loading model from cache #######')
    #     model = load_model(MODEL_FILE, custom_objects={'smooth_f2_score': metrics.smooth_f2_score,
    #                                                    'logloss_and_f2score': metrics.logloss_and_f2score})

    return model


def train(model):
    start = time.time()
    # 模型可视化，每一次保存会占用几秒钟
    tensorboard = keras.callbacks.TensorBoard(log_dir=BASE_DIR)
    checkpoint = keras.callbacks.ModelCheckpoint(filepath=SAVE_MODEL_FORMAT,
                                                 monitor="val_smooth_f2_score",
                                                 save_weights_only=True)
    model.fit(x_train, y_train,
              batch_size=TRAIN_BATCH_SIZE,
              epochs=EPOCH,
              verbose=1,
              validation_data=(x_valid, y_valid),
              callbacks=[tensorboard, checkpoint])
    print("####### train model spend %d seconds #######" % (time.time() - start))


def evaluate(model, files, x, y):
    from sklearn.metrics import fbeta_score

    y_pred = model.predict(x, batch_size=PREDICT_BATCH_SIZE)

    start = time.time()
    best_score, threshold = metrics.best_f2_score(y, y_pred)
    print("####### search threshold spend %d seconds ######" % (time.time() - start))

    print("####### Smooth F2-Score is %f #######" % metrics.smooth_f2_score_np(y, y_pred))
    print("####### F2-Score with threshold 0.2 is %f #######" % fbeta_score(y, (np.array(y_pred) > THRESHOLD).astype(
        np.int8), beta=2, average='samples'))
    print("####### Best F2-Score is %f #######" % best_score)

    y_pred_best = (np.array(y_pred) > threshold).astype(np.int8)
    with open(BASE_DIR + "evaluate.txt", "w+") as f:
        f.write("threadshold: ")
        f.write(",".join([str(j) for j in threshold]))
        f.write("\n")
        for i in range(len(files)):
            f.write(files[i])
            f.write(",")
            f.write(",".join([str(j) for j in list(y_pred_best[i])]))
            f.write("\n")


def evaluate_all(path, model, x, y):
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
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
                    "[%s]\t F2_smooth=%.4f,  F2_0.2=%.4f,  F2_basinhopping=%.4f\n" % (
                        weight_file, f2_smooth, f2_2, f2_basinhopping))


model = get_model()
# model.load_weights(MODEL_FILE)
train(model)
evaluate_all(BASE_DIR, model, x_valid, y_valid)
# evaluate(model, val_files, x_valid, y_valid)
# evaluate(model, train_files, x_train, y_train)
