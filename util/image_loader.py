import keras.preprocessing.image as kimage
import numpy as np

import util


def load_image(dictionary: str, image_size: tuple) -> kimage.NumpyArrayIterator:
    """
    导入一个路径下的所有图片，并生成可用直接训练的数据集

    :param dictionary: 图片所在文件夹
    :param image_size: 最终加载的图片尺寸，用于图片变形
    :return: 可直接用于keras模型训练的数据集
    """

    paths = kimage.list_pictures(dictionary)
    labels = []
    images = []
    for p in paths:
        images.append(kimage.img_to_array(kimage.load_img(p, target_size=image_size)))
        label = p.split(".")[-2].split("_")[1:]
        labels.append(list(map(int, label)))
    x = np.array(images)
    y = np.array(labels)

    # 暂时未进行任何的预处理
    gen = kimage.ImageDataGenerator()

    # batch size 32，自动shuffle，不导出
    return gen.flow(x, y)


if __name__ == '__main__':
    flow = load_image(util.downloader.DEMO_TRAINING_PHOTOS_PATH, (227, 227))
