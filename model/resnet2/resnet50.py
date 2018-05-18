import os

import tensorflow as tf

import config
from config import ModelConfig
from . import resnet_lib

MODEL_CONFIG = ModelConfig(k_fold_file="1.txt",
                           val_index=1,
                           image_size=(224, 224),
                           image_shape=(224, 224, 3),
                           data_type=[config.DATA_TYPE_ORIGINAL],
                           model_dir=os.path.dirname(os.path.abspath(__file__)),
                           record_sub_dir="1",
                           output_tensor_name="my_output/Sigmoid:0")


def get_model(image_shape):
    model = resnet_lib.ResNet50(include_top=False, weights='imagenet', input_shape=image_shape,
                                classes=1000)
    model.summary()

    # 去掉最后一个FC（Softmax层）
    # model.layers.pop()

    # 在原有的模型后面再添加一层，用于进行多标签分类
    output = tf.keras.layers.Dense(units=13, activation="sigmoid", name="my_output")(model.layers[-1].output)
    my_model = tf.keras.Model(model.input, output)
    my_model.summary()
    my_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return my_model


def get_estimator():
    model = get_model(MODEL_CONFIG.image_shape)

    estimator_config = tf.estimator.RunConfig(
        save_checkpoints_secs=10 * 60,
        keep_checkpoint_max=10000,
        save_summary_steps=50,
    )

    estimator = tf.keras.estimator.model_to_estimator(
        keras_model=model,
        model_dir=MODEL_CONFIG.record_dir,
        config=estimator_config
    )

    return estimator
