import math
import random

import keras.preprocessing.image as kimage
import numpy as np

import util


def devide_data(x: np.array, y: np.array, data_ratio=(0.8, 0.1, 0.1)) -> list:
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
            result.append((x[:pointer[i], :], y[:pointer[i], :]))
        else:
            result.append((x[pointer[i - 1]:pointer[i], :], y[pointer[i - 1]:pointer[i]]))

    return result


def load_image(dictionary, image_size=(227, 227)):
    """
    导入一个路径下的所有图片，并生成可用直接训练的数据集

    :param dictionary: 图片所在文件夹
    :param image_size: 最终加载的图片尺寸，用于图片变形
    :param batch_size: 批量尺寸
    :param data_split: 数据集划分，会根据给出的比例返回三个数据集
    :return: 可直接用于keras模型训练的数据集
    """

    paths = kimage.list_pictures(dictionary)
    random.shuffle(paths)

    labels = []
    images = []
    for p in paths:
        images.append(kimage.img_to_array(kimage.load_img(p, target_size=image_size)))
        label = p.split(".")[-2].split("_")[1:]
        labels.append(list(map(int, label)))
    x = np.array(images)
    y = np.array(labels)

    return x, y


if __name__ == '__main__':
    x, y = load_image(util.downloader.DEMO_TRAINING_PHOTOS_PATH, (227, 227))
    a, b = list(devide_data(x, y, (0.5, 0.5)))
    print(a, b)
