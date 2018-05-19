import tensorflow as tf


def sum_pred(y_true, y_pred):
    threshold = tf.constant(0.2)
    y_pred_b = tf.cast(tf.greater(y_pred, threshold), dtype=y_true.dtype)
    sum_pre = tf.reduce_sum(y_pred_b, axis=1)
    # tf.summary.scalar("train_sum_true", sum_pre)
    return sum_pre


def sum_true(y_true, y_pred):
    sum_tru = tf.reduce_sum(y_true, axis=1)
    # tf.summary.scalar("train_sum_true", sum_tru)
    return sum_tru


def sum_correct(y_true, y_pred):
    threshold = tf.constant(0.2)
    y_pred_b = tf.cast(tf.greater(y_pred, threshold), dtype=y_true.dtype)
    y_correct = y_true * y_pred_b
    sum_correc = tf.reduce_sum(y_correct, axis=1)
    # tf.summary.scalar("train_sum_correct", sum_correc)
    return sum_correc


def precision(y_true, y_pred):
    threshold = tf.constant(0.2)
    y_pred_b = tf.cast(tf.greater(y_pred, threshold), dtype=y_true.dtype)
    y_correct = y_true * y_pred_b
    sum_pred = tf.reduce_sum(y_pred_b, axis=1)
    sum_correct = tf.reduce_sum(y_correct, axis=1)
    precis = tf.reduce_mean(sum_correct / (sum_pred))
    # tf.summary.scalar("train_precision", precis)
    return precis


def recall(y_true, y_pred):
    threshold = tf.constant(0.2)
    y_pred_b = tf.cast(tf.greater(y_pred, threshold), dtype=y_true.dtype)
    y_correct = y_true * y_pred_b
    sum_true = tf.reduce_sum(y_true, axis=1)
    sum_correct = tf.reduce_sum(y_correct, axis=1)
    recal = tf.reduce_mean(sum_correct / (sum_true))
    # tf.summary.scalar("train_recall", recal)
    return recal


def f2_score(y_true, y_pred):
    threshold = tf.constant(0.2)
    y_pred_b = tf.cast(tf.greater(y_pred, threshold), dtype=y_true.dtype)
    y_correct = y_true * y_pred_b
    sum_true = tf.reduce_sum(y_true, axis=1)
    sum_pred = tf.reduce_sum(y_pred_b, axis=1)
    sum_correct = tf.reduce_sum(y_correct, axis=1)
    precision = sum_correct / (sum_pred)
    recall = sum_correct / (sum_true)
    f_score = 5 * precision * recall / (4 * precision + recall)
    f_score = tf.where(tf.is_nan(f_score), tf.zeros_like(f_score), f_score)
    f2 = tf.reduce_mean(f_score)
    # tf.summary.scalar("train_f2score", f2)
    return f2
