import math
import os
import random
import re

import numpy as np
import tensorflow as tf
import cv2

import config
from util import path


def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    #image_test = cv2.imread(filename.decode())
    image_decoded = tf.image.decode_jpeg(image_string)
    image_resize = tf.image.resize_images(image_decoded, config.IMAGE_SIZE)
    return image_resize, label

def get_labels(filenames):
    labels = []
    for i in filenames:
        label = i.split(".")[-2].split("_")[1:]
        labels.append(list(map(int, label)))
    return labels

def get_data_sets():
    train_files, val_files = get_k_fold_files("1.txt", 1, [config.DATA_TYPE_ORIGINAL])
    labels = get_labels(train_files)

    dataset = tf.data.Dataset.from_tensor_slices((train_files, labels))
    dataset = dataset.map(_parse_function)
    dataset = dataset.repeat(config.EPOCH)
    dataset = dataset.batch(config.BATCH_SIZE)
    dataset = dataset.prefetch(config.PREFETCH_BUFFER_SIZE)
    return dataset


def get_k_fold_files(k_fold_file, val_index, data_type: []):
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


def load_image(directory, image_size=(800, 800), train_set: bool = True):
    """
    导入一个路径下的所有图片，并生成可用直接训练的数据集
    图片名称
    :param directory: 图片所在文件夹
    :param image_size: 最终加载的图片尺寸，用于图片变形
    :param batch_size: 批量尺寸
    :param data_split: 数据集划分，会根据给出的比例返回三个数据集
    :return: 可直接用于keras模型训练的数据集
    """

    names = list_image_name(directory)

    random.shuffle(names)

    labels = []
    images = []
    for name in names:
        image_path = os.path.join(directory, name)
        images.append(
            tf.keras.preprocessing.image.img_to_array(
                tf.keras.preprocessing.image.load_img(image_path, target_size=image_size)))
        if train_set:
            # 训练数据的必须符合xx_xxx_label.xxx的格式
            label = name.split(".")[-2].split("_")[1:]
            labels.append(list(map(int, label)))
    x = np.array(images)
    y = np.array(labels)

    return x, y, np.array(names)


def unison_shuffled_copies(a, b):
    """
    将a， b，按照同种方式打乱（对应关系不变）
    :param a:
    :param b:
    :return:
    """

    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


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


def remove_image_original_header():
    names = list_image_name(path.ORIGINAL_TRAIN_IMAGES_PATH)
    for i in names:
        if i.split("_")[0] == "original":
            name_target = "_".join(i.split("_")[1:])
            os.rename(os.path.join(path.ORIGINAL_TRAIN_IMAGES_PATH, i),
                      os.path.join(path.ORIGINAL_TRAIN_IMAGES_PATH, name_target))


if __name__ == '__main__':
    # remove_image_original_header()
    get_data_sets()

