import tensorflow as tf

# 原生的keras中的resnet最小尺寸为197*197，对其进行了一点修改，去掉了合法性判断
# 且通过修改AveragePooling，支持在有全连接层的情况下也能修改图片输出尺寸
from model import resnet_lib


def get_model(image_shape):
    model = resnet_lib.ResNet50(include_top=False, weights='imagenet', input_shape=image_shape, classes=1000)
    model.summary()

    # 去掉最后一个FC（Softmax层）
    # model.layers.pop()

    # 在原有的模型后面再添加一层，用于进行多标签分类
    output = tf.keras.layers.Dense(units=13, activation="sigmoid", name="my_output")(model.layers[-1].output)
    my_model = tf.keras.Model(model.input, output)
    my_model.summary()

    my_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return my_model
