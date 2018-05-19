import math
import os
import random
import re

import cv2
import numpy as np
import tensorflow as tf

import config
from util import path


def _read_py_function(filename, label):
    image = cv2.imread(filename.decode())
    return image, label


def _resize_function(image_decoded, label):
    image_decoded.set_shape([None, None, 3])
    label.set_shape([13])
    image_resized = tf.image.resize_images(image_decoded, config.IMAGE_SIZE)
    return image_resized, label


def get_labels(filenames):
    labels = []
    for i in filenames:
        label = i.split(".")[-2].split("_")[1:]
        labels.append(list(map(int, label)))
    return labels


def data_input_fn(model_config: config.ModelConfig, validation=False):
    train_files, val_files = get_k_fold_files(model_config.k_fold_file, model_config.val_index, model_config.data_type)
    if validation:
        labels = get_labels(val_files)
        dataset = tf.data.Dataset.from_tensor_slices((val_files, labels))
    else:
        labels = get_labels(train_files)
        dataset = tf.data.Dataset.from_tensor_slices((train_files, labels))

    dataset = dataset.map(
        lambda filename, label: tuple(tf.py_func(
            _read_py_function, [filename, label], [tf.uint8, label.dtype])))
    dataset = dataset.map(_resize_function)
    if not validation:
        dataset = dataset.repeat(config.EPOCH)
    dataset = dataset.batch(config.BATCH_SIZE)
    dataset = dataset.prefetch(config.PREFETCH_BUFFER_SIZE)
    iterator = dataset.make_one_shot_iterator()

    features, labels = iterator.get_next()
    print(labels.name)
    return features, labels


def get_k_fold_files(k_fold_file, val_index, data_type: [], shuffle=True):
    train_names = []
    val_names = []
    with open(os.path.join(path.K_FOLD_TXT_PATH, k_fold_file), 'r') as f:
        for l in f.readlines():
            k, name = l.split(",")
            val_names.append(name.strip()) if int(k) is val_index else train_names.append(name.strip())

    train_files = []
    val_files = []

    for data in data_type:
        for name in train_names:
            train_files.append(os.path.join(path.get_train_data_path(data), name))
        for name in val_names:
            val_files.append(os.path.join(path.get_train_data_path(data), name))
    if shuffle:
        random.shuffle(train_files)
        random.shuffle(val_files)
    return train_files, val_files


def list_image_dir(directory, ext='jpg|jpeg|bmp|png|ppm'):
    """
    列出目录下的所有图片的路径
    :param directory:
    :param ext:
    :return:
    """
    return [os.path.join(root, f)
            for root, _, files in os.walk(directory) for f in files
            if re.match(r'([\w]+\.(?:' + ext + '))', f)]


def list_image_name(directory, ext='jpg|jpeg|bmp|png|ppm'):
    """
    列出目录下的所有图片的名称
    :param directory:
    :param ext:
    :return:
    """
    return [f for root, _, files in os.walk(directory)
            for f in files if re.match(r'([\w]+\.(?:' + ext + '))', f)]


def load_label(directory):
    """
    导入指定目录的所有图片的标签，不导入图片
    :param directory:
    :return:
    """
    names = list_image_name(directory)
    random.shuffle(names)
    labels = []
    for name in names:
        label = name.split(".")[-2].split("_")[1:]
        labels.append(list(map(int, label)))
    return np.array(labels), np.array(names)


def divide_data(x: np.array, y: np.array, data_ratio=(0.8, 0.1, 0.1)) -> list:
    """
    将数据根据比例分为N份，x和y的对应关系保持不变
    :param x:
    :param y:
    :param data_ratio: 划分比例，要求比例之和为1，划分次数不做限制
    :return:
    """

    assert sum(data_ratio) == 1

    data_num = x.shape[0]

    pointer = []
    ratio_sum = 0
    for ratio in data_ratio:
        ratio_sum += ratio
        pointer.append(math.floor(data_num * ratio_sum))

    result = []
    for i in range(len(pointer)):
        if i is 0:
            result.append((x[:pointer[i]], y[:pointer[i]]))
        else:
            result.append((x[pointer[i - 1]:pointer[i]], y[pointer[i - 1]:pointer[i]]))

    return result


def remove_image_name_header(dir):
    names = list_image_name(dir)
    for i in names:
        headr = i.split("_")[0]
        if headr == config.DATA_TYPE_SEGMENTED or headr == config.DATA_TYPE_AUGMENTED or headr == config.DATA_TYPE_ORIGINAL:
            name_target = "_".join(i.split("_")[1:])
            os.rename(os.path.join(dir, i),
                      os.path.join(dir, name_target))


if __name__ == '__main__':
    remove_image_name_header(path.ORIGINAL_TRAIN_IMAGES_PATH)
    remove_image_name_header(path.SEGMENTED_TRAIN_IMAGES_PATH)

    # data_input_fn()
