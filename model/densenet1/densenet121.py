import os

import tensorflow as tf

import config
from config import ModelConfig
from util import estimator
from util import metrics
from util import data_loader

DESCRIPTION = "DenseNet121，Include_top为False， 新增Dense1024-ReLu, Dense-13-sigmoid，使用加权Bce作为Loss"

MODEL_CONFIG = ModelConfig(DESCRIPTION,
                           k_fold_file="1.txt",
                           val_index=1,
                           image_size=(224, 224),
                           image_shape=(224, 224, 3),
                           data_type=[config.DATA_TYPE_SEGMENTED],
                           model_dir=os.path.dirname(os.path.abspath(__file__)),
                           record_sub_dir="1_3",
                           output_tensor_name="my_output/Sigmoid:0",
                           epoch=10,
                           batch_size=32,
                           learning_rate=0.0002)


def get_model(image_shape):
    base_model = tf.keras.applications.densenet.DenseNet121(include_top=False, input_shape=image_shape,
                                                            weights='imagenet')
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    predictions = tf.keras.layers.Dense(units=13, activation="sigmoid", name="my_output")(x)
    MODEL_CONFIG.output_tensor_name = predictions.name
    print("output tensor name is %s" % MODEL_CONFIG.output_tensor_name)

    model = tf.keras.Model(base_model.input, predictions)

    model.compile(loss=metrics.weighted_bce,
                  optimizer=tf.keras.optimizers.SGD(lr=MODEL_CONFIG.learning_rate, momentum=0.9),
                  metrics=["accuracy", metrics.sum_pred, metrics.sum_true, metrics.sum_correct, metrics.precision, metrics.recall,
                           metrics.smooth_f2_score])

    model.summary()

    return model


def get_estimator():
    model = get_model(MODEL_CONFIG.image_shape)

    estimator_config = tf.estimator.RunConfig(
        save_checkpoints_secs=20 * 60,
        keep_checkpoint_max=10000,
        save_summary_steps=100,
    )

    estimator = tf.keras.estimator.model_to_estimator(
        keras_model=model,
        model_dir=MODEL_CONFIG.record_dir,
        config=estimator_config
    )

    return estimator


if __name__ == "__main__":
    estimator.train_evaluate(get_estimator(), MODEL_CONFIG)
