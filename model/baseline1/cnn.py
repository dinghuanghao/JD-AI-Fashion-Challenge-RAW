import os
import time

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
CACHE_FILE = './record/001-baseline-model.h5'
THRESHOLD = 0.2
BATCH_SIZE = 64

train_files, val_files = data_loader.get_k_fold_files("baseline.txt", 1, [config.DATA_TYPE_ORIGINAL])

# # 取少量数据看模型是否run的起来
# train_files = train_files[:512]
# val_files = val_files[:128]

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

# 模型可视化，每一次保存会占用几秒钟
tensorboard = keras.callbacks.TensorBoard(log_dir='./record',
                                          histogram_freq=1,
                                          write_graph=True,
                                          write_images=False)

if os.path.isfile(CACHE_FILE):
    print('####### Loading model from cache ######')
    model = load_model(CACHE_FILE, custom_objects={'smooth_f2_score': metrics.smooth_f2_score})

start = time.time()

model.fit(x_train, y_train,
          batch_size=64,
          epochs=20,  # Should implement early stopping
          verbose=1,
          validation_data=(x_valid, y_valid),
          callbacks=[tensorboard])

print("####### train model spend %d seconds ######" % (time.time() - start))

model.save(CACHE_FILE)

from sklearn.metrics import fbeta_score

p_valid = model.predict(x_valid, batch_size=128)
print("####### F2-Score is %f ######" % fbeta_score(y_valid, np.array(p_valid) > THRESHOLD, beta=2, average='samples'))

######## Prediction ########

y_pred = model.predict(x_valid, batch_size=128)
y_pred_bool = (np.array(y_pred) > THRESHOLD).astype(np.int8)

with open("./record/predict.txt", "w+") as f:
    for i in range(len(val_files)):
        f.write(val_files[i])
        f.write(",")
        f.write(",".join([str(j) for j in list(y_pred_bool[i])]))
        f.write("\n")
