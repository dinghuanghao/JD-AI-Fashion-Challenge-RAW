import tensorflow as tf


def resnet50():
    model = tf.keras.applications.resnet50.ResNet50(include_top=True, weights='imagenet', classes=1000)
    model.summary()
    model.layers.pop()

    output = tf.keras.layers.Dense(units=13, activation="sigmoid", name="my_output")(model.layers[-1].output)
    my_model = tf.keras.Model(model.input, output)
    my_model.summary()

    my_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return my_model
