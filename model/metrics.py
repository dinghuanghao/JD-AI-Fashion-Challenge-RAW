import tensorflow as tf


def fbeta_score(precision, recall, beta):
    """
    计算FBeta-Score
    :param precision: 精确率
    :param recall: 召回率
    :param beta: 系数
    :return:
    """
    return (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)


def f2_score(y_true, y_pred):
    """
    计算F2-Score，要求y_true 和 y_pred 都是 0/1-original，而非概率
    :param y_true: 实际的标签
    :param y_pred: 预测的标签
    :return:
    """
    TP = tf.count_nonzero(y_pred * y_true)
    TN = tf.count_nonzero((y_pred - 1) * (y_true - 1))
    FP = tf.count_nonzero(y_pred * (y_true - 1))
    FN = tf.count_nonzero((y_pred - 1) * y_true)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    return fbeta_score(precision, recall, 2)
