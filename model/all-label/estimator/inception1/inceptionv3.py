import os

import tensorflow as tf

import config
from config import ModelConfig
from util import estimator
from util import metrics

DESCRIPTION = "InceptionV3，IncludeTop为False，预训练参数不参与训练， 新增Dense128-ReLu, Dense13-sigmoid，使用BCE作为Loss"

MODEL_CONFIG = ModelConfig(DESCRIPTION,
                           k_fold_file="1.txt",
                           val_index=1,
                           image_size=(224, 224),
                           image_shape=(224, 224, 3),
                           data_type=[config.DATA_TYPE_SEGMENTED],
                           model_dir=os.path.dirname(os.path.abspath(__file__)),
                           record_sub_dir="1_1",
                           output_tensor_name="my_output/Sigmoid:0",
                           epoch=10,
                           train_batch_size=32,
                           learning_rate=0.001)


def get_model(image_shape):
    # Create own input format
    model_input = tf.keras.layers.Input(shape=MODEL_CONFIG.image_shape, name='image_input')

    # Load Inception v3
    base_model = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False)
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model(model_input)
    feat = tf.keras.layers.Flatten(name='flatten')(x)
    feat = tf.keras.layers.Dense(128, activation='relu')(feat)
    feat = tf.keras.layers.BatchNormalization()(feat)
    out = tf.keras.layers.Dense(units=13, activation="sigmoid", name="my_output")(feat)
    model = tf.keras.Model(inputs=model_input, outputs=out)

    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.SGD(lr=MODEL_CONFIG.learning_rate, momentum=0.9),
                  metrics=['accuracy', metrics.sum_pred, metrics.sum_true, metrics.sum_correct, metrics.precision,
                           metrics.recall, metrics.smooth_f2_score])

    print('######## Summary ########')
    model.summary()

    return model


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
