import keras.preprocessing.image as kimage

import util
from model import alexnet

DEMO_TRAINING_PHOTOS_PATH = "./data/demo/photos/train"
DEMO_TEST_PHOTOS_PATH = "./data/demo/photos/test"
IMAGE_SIZE = (227, 227)
IMAGE_SHAPE = (227, 227, 3)
BATCH_SIZE = 32
EPOCHS = 2

model = alexnet.alexnet(IMAGE_SHAPE)
x, y = util.image_loader.load_image(DEMO_TRAINING_PHOTOS_PATH, IMAGE_SIZE)
(x_train, y_train), (x_val, y_val), (x_test, y_test) = util.image_loader.devide_data(x, y, (0.6, 0.2, 0.2))

flow_train = kimage.ImageDataGenerator().flow(x_train, y_train, batch_size=32)
flow_val = kimage.ImageDataGenerator().flow(x_val, y_val, batch_size=32)
flow_test = kimage.ImageDataGenerator().flow(x_test, y_test, batch_size=1)

model.fit_generator(flow_train, epochs=EPOCHS, validation_data=flow_val)
loss, accuracy = model.evaluate_generator(flow_test)
print("loss is %f, accuracy is %f" % (loss, accuracy))

x, _ = util.image_loader.load_image(DEMO_TEST_PHOTOS_PATH, IMAGE_SIZE)
predicts = model.predict(x, batch_size=1, verbose=1)
print(predicts)
