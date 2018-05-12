import keras.preprocessing.image as kimage

import util
from model import alexnet
from model import resnet50

DEMO_TRAINING_PHOTOS_PATH = "./data/demo/photos/train"
DEMO_TEST_PHOTOS_PATH = "./data/demo/photos/test"
IMAGE_SIZE = (224, 224)
IMAGE_SHAPE = (224, 224, 3)
BATCH_SIZE = 32
EPOCHS = 5

#model = alexnet.alexnet(IMAGE_SHAPE)
model = resnet50.resnet50()
x, y = util.image_loader.load_image(DEMO_TRAINING_PHOTOS_PATH, IMAGE_SIZE)
(x_train, y_train), (x_val, y_val), (x_test, y_test) = util.image_loader.devide_data(x, y, (0.6, 0.2, 0.2))

flow_train = kimage.ImageDataGenerator().flow(x_train, y_train, batch_size=32)
flow_val = kimage.ImageDataGenerator().flow(x_val, y_val, batch_size=32)
flow_test = kimage.ImageDataGenerator().flow(x_test, y_test, batch_size=1)

#TODO: 改为minibatch训练方式，一定批次进行一次阈值搜索，并判断F2-SCORE以及模型的保存
model.fit_generator(flow_train, steps_per_epoch=len(flow_train)/BATCH_SIZE, epochs=EPOCHS, validation_data=flow_val, validation_steps=len(flow_val)/BATCH_SIZE)
loss, metrics = model.evaluate_generator(flow_test)
print("loss is %f, metrics is %f" % (loss, metrics))

x, _ = util.image_loader.load_image(DEMO_TEST_PHOTOS_PATH, IMAGE_SIZE)
predicts = model.predict(x, batch_size=1, verbose=1)
print(predicts)
