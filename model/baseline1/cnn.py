import os

import cv2
import keras
import numpy as np
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten, BatchNormalization
from keras.models import Sequential, load_model
from tqdm import tqdm

import config
from util import data_loader
from util import metrics

RESOLUTION = 128
CACHE_FILE = './record/weights.53-0.78.hdf5'
THRESHOLD = 0.2
BATCH_SIZE = 64

train_files, val_files = data_loader.get_k_fold_files("baseline.txt", 1, [config.DATA_TYPE_ORIGINAL])

# 取少量数据看模型是否run的起来
train_files = train_files[:32]
val_files = val_files[:128]

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

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy', metrics.smooth_f2_score])

    if os.path.isfile(CACHE_FILE):
        print('####### Loading model from cache ######')
        model = load_model(CACHE_FILE, custom_objects={'smooth_f2_score': metrics.smooth_f2_score})

    return model


def train(model):
    import time
    start = time.time()
    # 模型可视化，每一次保存会占用几秒钟
    tensorboard = keras.callbacks.TensorBoard(log_dir='./record',
                                              histogram_freq=1,
                                              write_graph=True,
                                              write_images=False)
    checkpoint = keras.callbacks.ModelCheckpoint(filepath="./record/weights.{epoch:02d}-{val_smooth_f2_score:.2f}.hdf5",
                                                 monitor="val_smooth_f2_score",
                                                 verbose=0,
                                                 save_best_only=True,
                                                 save_weights_only=False,
                                                 mode="max",
                                                 period=1)
    model.fit(x_train, y_train,
              batch_size=32,
              epochs=200,  # Should implement early stopping
              verbose=1,
              validation_data=(x_valid, y_valid),
              callbacks=[tensorboard, checkpoint])
    print("####### train model spend %d seconds ######" % (time.time() - start))
    model.save(CACHE_FILE)


def predict(model, files, x, y):
    from sklearn.metrics import fbeta_score

    y_pred = model.predict(x, batch_size=128)
    best_score, threshold = metrics.best_f2_score(y, y_pred)

    print("####### Smooth F2-Score is %f #######" % metrics.smooth_f2_score_np(y, y_pred))
    print("####### F2-Score with threshold 0.2 is %f #######" % fbeta_score(y, (np.array(y_pred) > THRESHOLD).astype(
        np.int8), beta=2, average='samples'))
    print("####### Best F2-Score is %f #######" % best_score)

    y_pred_best = (np.array(y_pred) > threshold).astype(np.int8)
    with open("./record/predict.txt", "w+") as f:
        f.write("threadshold: ")
        f.write(",".join([str(j) for j in threshold]))
        f.write("\n")
        for i in range(len(files)):
            f.write(files[i])
            f.write(",")
            f.write(",".join([str(j) for j in list(y_pred_best[i])]))
            f.write("\n")


model = get_model()
# train(model)
predict(model, val_files, x_valid, y_valid)
# predict(model, train_files, x_train, y_train)
