import os

import tensorflow as tf

import config
from config import ModelConfig
from util import estimator
from util import metrics

DESCRIPTION = "DenseNet121，Include_top为False， 新增Dense1024-ReLu, Dense-13-sigmoid，使用加权Bce作为Loss"

MODEL_CONFIG = ModelConfig(DESCRIPTION,
                           k_fold_file="1.txt",
                           val_index=1,
                           image_size=(224, 224),
                           image_shape=(224, 224, 3),
                           data_type=[config.DATA_TYPE_SEGMENTED],
                           model_dir=os.path.dirname(os.path.abspath(__file__)),
                           record_sub_dir="1_3_1",
                           output_tensor_name="my_output/Sigmoid:0",
                           epoch=10,
                           batch_size=32,
                           learning_rate=0.0001)


def get_model(image_shape):
    base_model = tf.keras.applications.densenet.DenseNet121(include_top=False, input_shape=image_shape,
                                                            weights='imagenet')
    for layer in base_model.layers:
        layer.trainable = False

    # 对最后两个Block进行fine tune
    for layer in base_model.layers[-15:]:
        layer.trainable = True

    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    predictions = tf.keras.layers.Dense(units=13, activation="sigmoid", name="my_output")(x)
    MODEL_CONFIG.output_tensor_name = predictions.name
    print("output tensor name is %s" % MODEL_CONFIG.output_tensor_name)

    model = tf.keras.Model(base_model.input, predictions)

    model.compile(loss=metrics.weighted_bce,
                  optimizer=tf.keras.optimizers.SGD(lr=MODEL_CONFIG.learning_rate, momentum=0.9),
                  metrics=["accuracy", metrics.sum_pred, metrics.sum_true, metrics.sum_correct,
                           metrics.precision, metrics.recall, metrics.smooth_f2_score,
                           metrics.average_1, metrics.average_2, metrics.average_3, metrics.average_4,
                           metrics.average_5, metrics.average_6, metrics.average_7, metrics.average_8,
                           metrics.average_9, metrics.average_10, metrics.average_11, metrics.average_12])

    model.summary()

    return model


def get_estimator():
    model = get_model(MODEL_CONFIG.image_shape)

    # train_and_evaluate()会每个epoch评估并保存一次，为提高性能，此处设置为非常大的保存周期
    estimator_config = tf.estimator.RunConfig(
        save_checkpoints_secs=2000 * 60,
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
    # estimator.predict(get_estimator(), MODEL_CONFIG)

    # data_loader.read_and_save_checkpoint(os.path.join(MODEL_CONFIG.record_dir, "keras_model.ckpt"),
    #                                      os.path.join(MODEL_CONFIG.record_dir, "keras_model_checkpoint.txt"))
