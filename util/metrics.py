import keras.backend as K
import numpy as np
import tensorflow as tf
import keras

weight_matrix = np.array(([[527, 12.8, 1.1, 210, 2.8, 6.18, 279.32, 40.5, 1.11, 7.7, 14.79, 43.9, 156]]),
                         dtype=np.float32)


def weighted_bce(y_true, y_pred):
    # 当标签的输出为1时，根据权重矩阵对其loss进行放大，当其为0时，权重为1。因为大量样本都是0，对其进行放大会导致更接近0
    weight = y_true * weight_matrix
    weight = tf.clip_by_value(weight, 1, np.max(weight_matrix))

    bce = K.binary_crossentropy(y_true, y_pred)
    print(bce.shape)
    bce_wighted = bce * weight
    return K.mean(bce_wighted, axis=-1)


def average_1(y_true, y_pred):
    return tf.reduce_mean(y_pred[:, 1])


def average_2(y_true, y_pred):
    return tf.reduce_mean(y_pred[:, 2])


def average_3(y_true, y_pred):
    return tf.reduce_mean(y_pred[:, 3])


def average_4(y_true, y_pred):
    return tf.reduce_mean(y_pred[:, 4])


def average_5(y_true, y_pred):
    return tf.reduce_mean(y_pred[:, 5])


def average_6(y_true, y_pred):
    return tf.reduce_mean(y_pred[:, 6])


def average_7(y_true, y_pred):
    return tf.reduce_mean(y_pred[:, 7])


def average_8(y_true, y_pred):
    return tf.reduce_mean(y_pred[:, 8])


def average_9(y_true, y_pred):
    return tf.reduce_mean(y_pred[:, 9])


def average_10(y_true, y_pred):
    return tf.reduce_mean(y_pred[:, 10])


def average_11(y_true, y_pred):
    return tf.reduce_mean(y_pred[:, 11])


def average_12(y_true, y_pred):
    return tf.reduce_mean(y_pred[:, 12])


def average_13(y_true, y_pred):
    return tf.reduce_mean(y_pred[:, 13])


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


def smooth_f2_score_np(y_true: np.ndarray, y_pred: np.ndarray, epsilon=1e-9):
    y_true = y_true.astype(y_pred.dtype)
    y_correct = y_true * y_pred
    sum_true = np.sum(y_true, axis=-1)
    sum_pred = np.sum(y_pred, axis=-1)
    sum_correct = np.sum(y_correct, axis=-1)
    precision = sum_correct / (sum_pred + epsilon)
    recall = sum_correct / (sum_true + epsilon)
    f2_score = 5 * precision * recall / (4 * precision + recall + epsilon)
    f2_score = np.where(np.isnan(f2_score), np.zeros_like(f2_score), f2_score)
    return np.mean(f2_score)


def smooth_f2_score_02(y_true, y_pred):
    tp = y_pred.dtype
    y_pred = y_pred > 0.2
    y_true = tf.cast(y_true, tp)
    y_pred = tf.cast(y_pred, tp)
    y_correct = y_true * y_pred
    sum_true = tf.reduce_sum(y_true, axis=-1)
    sum_pred = tf.reduce_sum(y_pred, axis=-1)
    sum_correct = tf.reduce_sum(y_correct, axis=-1)
    precision = sum_correct / (sum_pred + K.epsilon())
    recall = sum_correct / (sum_true + K.epsilon())
    f_score = 5 * precision * recall / (4 * precision + recall + K.epsilon())
    f_score = tf.where(tf.is_nan(f_score), tf.zeros_like(f_score), f_score)
    return tf.reduce_mean(f_score)


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


import numpy as np
import logging
from sklearn.metrics import fbeta_score
from scipy.optimize import basinhopping
from timeit import default_timer as timer

## Get the same logger from main"
logger = logging.getLogger("Planet-Amazon")


def best_f2_score(true_labels, predictions, label_num=13):
    def f_neg(threshold):
        ## Scipy tries to minimize the function so we must get its inverse
        if label_num > 1:
            return - fbeta_score(true_labels, predictions > threshold, beta=2, average='samples')
        else:
            return - fbeta_score(true_labels, predictions > threshold, beta=2)

    # Initialization of best threshold search
    thr_0 = [0.20] * label_num
    constraints = [(0., 1.)] * label_num

    def bounds(**kwargs):
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= 1))
        tmin = bool(np.all(x >= 0))
        return tmax and tmin

    # Search using L-BFGS-B, the epsilon step must be big otherwise there is no gradient
    minimizer_kwargs = {"method": "L-BFGS-B",
                        "bounds": constraints,
                        "options": {
                            "eps": 0.05
                        }
                        }

    # We combine L-BFGS-B with Basinhopping for stochastic search with random steps
    logger.info("===> Searching optimal threshold for each label")
    start_time = timer()

    opt_output = basinhopping(f_neg, thr_0,
                              stepsize=0.1,
                              minimizer_kwargs=minimizer_kwargs,
                              niter=10,
                              accept_test=bounds)

    end_time = timer()
    logger.info("===> Optimal threshold for each label:\n{}".format(opt_output.x))
    logger.info("Threshold found in: %s seconds" % (end_time - start_time))

    score = - opt_output.fun
    return score, opt_output.x


def greedy_f2_score(y_true, y_pred, label_num=13):
    print("label number is %d" % label_num)
    threshold = [0.10] * label_num
    best_score = 0
    best_threshold = [t for t in threshold]

    for i in range(label_num):
        threshold = [t for t in best_threshold]
        for j in range(100):
            threshold[i] = j / 100.
            if label_num > 1:
                score = fbeta_score(y_true, (np.array(y_pred) > threshold).astype(np.int8), beta=2, average='samples')
            else:
                score = fbeta_score(y_true, (np.array(y_pred) > threshold).astype(np.int8), beta=2)
            if score > best_score:
                best_score = score
                best_threshold[i] = threshold[i]

    return best_score, best_threshold
