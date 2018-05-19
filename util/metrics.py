import keras.backend as K
import tensorflow as tf


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
