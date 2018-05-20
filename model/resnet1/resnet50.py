import os

import tensorflow as tf

import config
from config import ModelConfig
from util import estimator
from util import metrics

MODEL_CONFIG = ModelConfig(k_fold_file="1.txt",
                           val_index=1,
                           image_size=(224, 224),
                           image_shape=(224, 224, 3),
                           data_type=[config.DATA_TYPE_SEGMENTED],
                           model_dir=os.path.dirname(os.path.abspath(__file__)),
                           record_sub_dir="1_3",
                           output_tensor_name="my_output/Sigmoid:0",
                           epoch=40,
                           train_batch_size=32,
                           learning_rate=0.0001)


def get_model(image_shape):
    model = tf.keras.applications.resnet50.ResNet50(include_top=True, input_shape=image_shape, weights='imagenet',
                                                    classes=1000)
    model.summary()
    model.layers.pop()

    optimizer = tf.keras.optimizers.Adam(lr=MODEL_CONFIG.learning_rate)
    output = tf.keras.layers.Dense(units=13, activation="sigmoid", name="my_output")(model.layers[-1].output)
    my_model = tf.keras.Model(model.input, output)
    my_model.summary()

    def logloss_and_f2score(p_true, p_pred):
        return tf.keras.losses.binary_crossentropy(p_true, p_pred) + metrics.f2_score_loss(p_true, p_pred)

    my_model.compile(loss=logloss_and_f2score, optimizer=optimizer,
                     metrics=[metrics.sum_pred, metrics.sum_true, metrics.sum_correct, metrics.precision,
                              metrics.recall, metrics.smooth_f2_score])
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


if __name__ == "__main__":
    estimator.train_evaluate(get_estimator(), MODEL_CONFIG)
