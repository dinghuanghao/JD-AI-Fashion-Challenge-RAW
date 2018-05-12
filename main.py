import keras.preprocessing.image as kimage

import util
from model import alexnet
from model import resnet50
from model import vgg16

DEMO_TRAINING_PHOTOS_PATH = "./data/demo/photos/train"
DEMO_TEST_PHOTOS_PATH = "./data/demo/photos/test"
IMAGE_SIZE = (48, 48)
IMAGE_SHAPE = (48, 48, 3)
BATCH_SIZE = 32
EPOCHS = 50

#model = get_model.get_model(IMAGE_SHAPE)
#model = vgg16.get_model()
model = resnet50.get_model(IMAGE_SHAPE)

x, y = util.image_loader.load_image(DEMO_TRAINING_PHOTOS_PATH, IMAGE_SIZE)
(x_train, y_train), (x_val, y_val), (x_test, y_test) = util.image_loader.devide_data(x, y, (0.6, 0.2, 0.2))

flow_train = kimage.ImageDataGenerator().flow(x_train, y_train, batch_size=32)
flow_val = kimage.ImageDataGenerator().flow(x_val, y_val, batch_size=32)
flow_test = kimage.ImageDataGenerator().flow(x_test, y_test, batch_size=1)

#TODO: 改为minibatch训练方式，一定批次进行一次阈值搜索，并判断F2-SCORE以及模型的保存
model.fit_generator(flow_train, steps_per_epoch=len(flow_train),
                    epochs=EPOCHS, validation_data=flow_val,
                    validation_steps=len(flow_val))
loss, metrics = model.evaluate_generator(flow_test, steps=len(flow_test))
print("loss is %f, metrics is %f" % (loss, metrics))

#此处仅仅是用来对比一下，看看效果，上面的evaluate_generator已经对test set进行了评估
#总体来说，因为标签比较稀疏，导致倾向于输出0来使得accuracy变高，需要想办法解决Class-Imbalance问题!!!
batch_num = 0
for test_batch_x, test_batch_y  in flow_test:
    predicts = model.predict(test_batch_x, batch_size=1, verbose=1)
    batch_num += 1
    print("start %d test: " % batch_num)
    print("    true:", "".join(str(e) for e in test_batch_y))
    print("    pred:", "".join(str(e) for e in predicts))


#DEMO中的测试数据集是没有标签的，只能进行预测
x, _ = util.image_loader.load_image(DEMO_TEST_PHOTOS_PATH, IMAGE_SIZE)

predicts = model.predict(x, batch_size=1, verbose=1)
print(predicts)
