import tensorflow as tf


def get_model():
    # 由于内存的缘故，去掉最后的几个FC层（include_top=False）
    model = tf.keras.applications.vgg16.VGG16(include_top=False, input_shape=(48, 48, 3), classes=1000)
    x = model.layers[-1].output
    x = tf.keras.layers.Flatten(name='flatten')(x)

    # 在原有的模型后面再添加一层，用于进行多标签分类
    output = tf.keras.layers.Dense(units=13, activation="sigmoid", name="my_output")(x)
    my_model = tf.keras.Model(model.input, output)
    my_model.summary()

    my_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return my_model
