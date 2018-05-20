import keras.backend as K
import numpy as np
import tensorflow as tf

weight_matrix = np.array(([[527, 12.8, 1.1, 210, 2.8, 6.18, 279.32, 40.5, 1.11, 7.7, 14.79, 43.9, 156]]), dtype=np.float32)


def weighted_bce(y_true, y_pred):
    #当标签的输出为1时，根据权重矩阵对其loss进行放大，当其为0时，权重为1。因为大量样本都是0，对其进行放大会导致更接近0
    weight = y_true * weight_matrix
    weight = tf.clip_by_value(weight, 1, np.max(weight_matrix))

    bce = K.binary_crossentropy(y_true, y_pred)
    print(bce.shape)
    bce_wighted = bce*weight
    return K.mean(bce_wighted, axis=-1)


def sum_pred(y_true, y_pred):
    return tf.reduce_sum(y_pred, axis=-1)


def sum_true(y_true, y_pred):
    return tf.reduce_sum(y_true, axis=-1)


def sum_correct(y_true, y_pred):
    y_correct = y_true * y_pred
    return tf.reduce_sum(y_correct, axis=-1)


def precision(y_true, y_pred):
    y_correct = y_true * y_pred
    sum_pred = tf.reduce_sum(y_pred, axis=-1)
    sum_correct = tf.reduce_sum(y_correct, axis=-1)
    precis = tf.reduce_mean(sum_correct / (sum_pred + K.epsilon()))
    return precis


def recall(y_true, y_pred):
    y_correct = y_true * y_pred
    sum_true = tf.reduce_sum(y_true, axis=-1)
    sum_correct = tf.reduce_sum(y_correct, axis=-1)
    recal = tf.reduce_mean(sum_correct / (sum_true + K.epsilon()))
    return recal


def f2_score_loss(y_true, y_pred):
    return 1 - smooth_f2_score(y_true, y_pred)


def smooth_f2_score(y_true, y_pred):
    y_true = tf.cast(y_true, y_pred.dtype)
    y_correct = y_true * y_pred
    sum_true = tf.reduce_sum(y_true, axis=-1)
    sum_pred = tf.reduce_sum(y_pred, axis=-1)
    sum_correct = tf.reduce_sum(y_correct, axis=-1)
    precision = sum_correct / (sum_pred + K.epsilon())
    recall = sum_correct / (sum_true + K.epsilon())
    f_score = 5 * precision * recall / (4 * precision + recall + K.epsilon())
    f_score = tf.where(tf.is_nan(f_score), tf.zeros_like(f_score), f_score)
    return tf.reduce_mean(f_score)


def logloss_and_f2score(p_true, p_pred):
    return tf.keras.losses.binary_crossentropy(p_true, p_pred) + f2_score_loss(p_true, p_pred)
