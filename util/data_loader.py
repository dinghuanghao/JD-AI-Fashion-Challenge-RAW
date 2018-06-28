import math
import os
import pathlib
import random
import re

import cv2
import keras.backend as K
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, Iterator, img_to_array
from tqdm import tqdm

import config
from util import path

training_times = 0
validation_times = 0


class TestTimeAugmentation():
    def __init__(self,
                 shift_range=0.1,
                 flip=True,
                 crop=True):

        if crop and flip:
            # 四个角落 + 中心 + 对称， 10-fold tta
            print("use crop and flip tta")
            self.width_shift_range = [
                [0, 1 - shift_range],
                [0, 1 - shift_range],
                [shift_range, 1],
                [shift_range, 1],
                [shift_range / 2, 1 - shift_range / 2],
                [0, 1 - shift_range],
                [0, 1 - shift_range],
                [shift_range, 1],
                [shift_range, 1],
                [shift_range / 2, 1 - shift_range / 2]
            ]

            self.height_shift_range = [
                [0, 1 - shift_range],
                [shift_range, 1],
                [0, 1 - shift_range],
                [shift_range, 1],
                [shift_range / 2, 1 - shift_range / 2],
                [0, 1 - shift_range],
                [shift_range, 1],
                [0, 1 - shift_range],
                [shift_range, 1],
                [shift_range / 2, 1 - shift_range / 2]
            ]

            self.horizontal_flip = [
                False,
                False,
                False,
                False,
                False,
                True,
                True,
                True,
                True,
                True
            ]
            self.tta_times = 10
        elif flip and not crop:
            print("only use flip tta")
            self.horizontal_flip = [
                False,
                True,
            ]
            self.width_shift_range = None
            self.height_shift_range = None
            self.tta_times = 2
        elif crop and not flip:
            self.width_shift_range = [
                [0, 1 - shift_range],
                [0, 1 - shift_range],
                [shift_range, 1],
                [shift_range, 1],
                [shift_range / 2, 1 - shift_range / 2]
            ]

            self.height_shift_range = [
                [0, 1 - shift_range],
                [shift_range, 1],
                [0, 1 - shift_range],
                [shift_range, 1],
                [shift_range / 2, 1 - shift_range / 2]
            ]
            self.horizontal_flip = None
            self.tta_times = 5


class KerasGenerator(ImageDataGenerator):
    def __init__(self, model_config=None, pca_jitter=False, pca_jitter_range=0.01, real_transform=False,
                 tta: TestTimeAugmentation = None, squre_crop=False, *args,
                 **kwargs):
        super(KerasGenerator, self).__init__(*args, **kwargs)
        self.iterator = None
        self.real_transform = real_transform
        self.pca_jitter = pca_jitter
        self.model_config = model_config
        self.pca_jitter_range = pca_jitter_range
        self.tta = tta
        self.squre_crop = squre_crop

        if model_config is not None:
            if not model_config.input_norm:
                self.featurewise_std_normalization = False
                self.featurewise_center = False
                self.rescale = None

            if self.featurewise_center or self.featurewise_std_normalization:
                self.check_mean_std_file(model_config)
                # 将check和load合并为load
                self.load_image_global_mean_std(model_config.image_mean_file, model_config.image_std_file)

    def flow_from_files(self, img_files, save_image_number=100,
                        mode='fit',
                        target_size=(256, 256),
                        batch_size=32,
                        tta_index=None,
                        shuffle=False, seed=None, label_position=None):
        return KerasIterator(self, img_files, save_image_number=100,
                             mode=mode,
                             target_size=target_size,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             seed=seed,
                             data_format=None,
                             tta_index=tta_index,
                             label_position=label_position)

    def check_mean_std_file(self, model_config):
        if not os.path.exists(model_config.image_std_file) or not os.path.exists(model_config.image_mean_file):
            self.calc_image_global_mean_std(model_config.train_files)
            self.save_image_global_mean_std(model_config.image_mean_file, model_config.image_std_file)

    def calc_image_global_mean_std(self, img_files):
        shape = (1, 3)
        mean = np.zeros(shape, dtype=np.float32)
        M2 = np.zeros(shape, dtype=np.float32)

        print('Computing mean and standard deviation on the dataset')
        for n, file in enumerate(tqdm(img_files, miniters=256), 1):
            img = cv2.imread(os.path.join(file)).astype(np.float32)
            img *= self.rescale
            mean_current = np.mean(img, axis=(0, 1)).reshape((1, 3))
            delta = mean_current - mean
            mean += delta / n
            delta2 = mean_current - mean
            M2 += delta * delta2

        self.mean = mean
        self.std = M2 / (len(img_files) - 1)

        print("Calc image mean: %s" % str([str(i) for i in self.mean]))
        print("Calc image std: %s" % str([str(i) for i in self.std]))

    def save_image_global_mean_std(self, path_mean, path_std):
        if self.mean is None or self.std is None:
            raise ValueError('Mean and Std must be computed before, fit the generator first')
        np.save(path_mean, self.mean.reshape((1, 3)))
        np.save(path_std, self.std.reshape((1, 3)))

    def load_image_global_mean_std(self, path_mean, path_std):
        self.mean = np.load(path_mean)
        self.std = np.load(path_std)
        print("Load image mean: %s" % str([str(i) for i in self.mean]))
        print("Load image std: %s" % str([str(i) for i in self.std]))


class KerasIterator(Iterator):
    def __init__(self, generator: KerasGenerator, img_files, save_image_number=100,
                 mode='fit',
                 target_size=(256, 256),
                 batch_size=32, shuffle=None, seed=None,
                 data_format=None,
                 tta_index=None,
                 label_position=None):

        self.target_size = tuple(target_size)

        if data_format is None:
            self.data_format = K.image_data_format()

        if self.data_format == 'channels_last':
            self.image_shape = self.target_size + (3,)
        else:
            self.image_shape = (3,) + self.target_size

        self.generator = generator
        self.save_image_number = save_image_number
        self.img_files = img_files
        self.mode = mode
        if label_position is None:
            self.labels = np.array(get_labels(img_files), dtype=np.int8)
        else:
            self.labels = np.array(get_labels(img_files), dtype=np.int8)[:, label_position]

        self.tta_index = tta_index

        if mode == 'fit':
            self.debug_img_trans(self.generator.model_config.fit_img_record_dir)
        elif mode == 'predict' and self.generator.tta is not None and self.tta_index is not None:
            print("save predict img")
            record_dir = os.path.join(self.generator.model_config.predict_img_record_dir,
                                      str(self.tta_index))
            pathlib.Path(record_dir).mkdir(parents=True, exist_ok=True)
            self.debug_img_trans(record_dir)

        # Init parent class
        super(KerasIterator, self).__init__(len(self.img_files), batch_size, shuffle, seed)

    def real_transform(self, img):
        if self.mode == "predict" and self.generator.tta is not None and self.tta_index is not None:
            if self.generator.tta.height_shift_range is not None and self.generator.tta.width_shift_range is not None:
                width_start = int(self.generator.tta.width_shift_range[self.tta_index][0] * img.shape[1])
                width_end = int(self.generator.tta.width_shift_range[self.tta_index][1] * img.shape[1])
                height_start = int(self.generator.tta.height_shift_range[self.tta_index][0] * img.shape[0])
                height_end = int(self.generator.tta.height_shift_range[self.tta_index][1] * img.shape[0])

                img = img[height_start:height_end, width_start:width_end]

            if self.generator.tta.horizontal_flip is not None:
                if self.generator.tta.horizontal_flip[self.tta_index]:
                    img = cv2.flip(img, 1)

            return img

        if self.generator.squre_crop:
            # 采用正方形裁剪，确保resize的时候不会变形，alex等论文中均是这种方式，但是实测效果并不太好
            if self.generator.width_shift_range > 0 and self.generator.height_shift_range > 0:
                shift = min(self.generator.width_shift_range, self.generator.height_shift_range)
                side_length = int(min(img.shape[0], img.shape[1]) * (1 - shift))

                width_start = int(random.uniform(0, shift) * img.shape[1])
                height_start = int(random.uniform(0, shift) * img.shape[0])

                img = img[height_start:height_start + side_length, :]
                img = img[:, width_start:width_start + side_length]

                assert img.shape[0] == img.shape[1]
            return img

        # 这种方式会有轻微的拉伸或者压缩，因为裁剪之后的图片长宽可能不等，但是反而效果要好一个百分点
        # 可能是存在潜在的图像缩放
        if self.generator.width_shift_range > 0:
            width_shift = int(self.generator.width_shift_range * 100)
            width_start = int(random.randrange(-width_shift, width_shift) / 100 * img.shape[1])
            if width_start < 0:
                img = img[:, :width_start, :]
            else:
                img = img[:, width_start:, :]

        if self.generator.height_shift_range > 0:
            height_shift = int(self.generator.height_shift_range * 100)
            height_start = int(random.randrange(-height_shift, height_shift) / 100 * img.shape[0])

            if height_start < 0:
                img = img[:height_start, :]
            else:
                img = img[height_start:, :]

        if self.generator.horizontal_flip:
            if np.random.random() < 0.5:
                img = cv2.flip(img, 1)

        if self.generator.vertical_flip:
            if np.random.random() < 0.5:
                img = cv2.flip(img, 0)

        return img

    def pca_jitter(self, img):
        img = np.asanyarray(img, dtype='float32')
        img_size = img.size / 3
        img1 = img.reshape(int(img_size), 3)
        img1 = np.transpose(img1)
        img_cov = np.cov([img1[0], img1[1], img1[2]])
        lamda, p = np.linalg.eig(img_cov)  # 计算矩阵特征向量

        p = np.transpose(p)

        alpha1 = random.normalvariate(0, self.generator.pca_jitter_range)  # 生成正态分布的随机数
        alpha2 = random.normalvariate(0, self.generator.pca_jitter_range)
        alpha3 = random.normalvariate(0, self.generator.pca_jitter_range)

        v = np.transpose((alpha1 * lamda[0], alpha2 * lamda[1], alpha3 * lamda[2]))  # 加入扰动
        add_num = np.dot(p, v)

        img[:, :, 0] = img[:, :, 0] + add_num[0]
        img[:, :, 1] = img[:, :, 1] + add_num[1]
        img[:, :, 2] = img[:, :, 2] + add_num[2]

        return img

    def image_transform(self, img):
        if not self.generator.real_transform:
            img = cv2.resize(img, self.target_size)
            x = img_to_array(img, data_format=self.data_format)
            x = self.generator.random_transform(x)
        else:
            img = self.real_transform(img)
            img = cv2.resize(img, self.target_size)
            x = img_to_array(img, data_format=self.data_format)

        x = self.generator.standardize(x)
        if self.generator.pca_jitter:
            x = self.pca_jitter(x)

        return x

    def debug_img_trans(self, dir):
        # 训练每一个模型的时候，都保存一定的图片，用于检查data augmentation的效果
        print("save image augmentation to %s" % dir)
        if self.save_image_number > 0:
            for file in self.generator.model_config.val_files[0][:self.save_image_number]:
                img = cv2.imread(file)
                try:
                    img_trans = self.image_transform(img)
                except:
                    print("image trans exception, %s" % file)
                img_rescale = img_trans + max(-np.min(img_trans), 0)
                x_max = np.max(img_rescale)
                if x_max != 0:
                    img_rescale /= x_max
                    img_rescale *= 255.0

                img_name = os.path.split(file)[-1]
                cv2.imwrite(os.path.join(dir, img_name), img_rescale)
            self.save_image_number = 0

    def _get_batches_of_transformed_samples(self, index_array):
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=K.floatx())

        # Build batch of images
        for i, j in enumerate(index_array):
            file = self.img_files[j]
            img = cv2.imread(file)
            batch_x[i] = self.image_transform(img)

        # Build batch of labels.
        if self.mode == 'fit':
            batch_y = self.labels[index_array]
            return batch_x, batch_y
        elif self.mode == 'predict':
            return batch_x
        else:
            raise ValueError('The mode should be either \'fit\' or \'predict\'')

    def next(self):
        """For python 2.x.

        # Returns
            The next batch.
        """
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)


def up_sampling(files: np.ndarray, label_position):
    """
    对某一个标签进行上采样
    :param files:
    :param label_position:
    :return:
    """
    assert len(label_position) == 1

    y = np.array(get_labels(files), np.bool)[:, label_position]

    y_1 = y[y == 1]
    y_0 = y[y == 0]

    assert y_1.size != 0 and y_0.size != 0

    if y_1.size == y_0.size:
        return files

    if y_1.size > y_0.size:
        file_less = files[(y == 1)[:, 0]]
        n = y_1.size // y_0.size
        m = y_1.size % y_0.size

    elif y_1.size < y_0.size:
        file_less = files[(y == 1)[:, 0]]
        n = y_0.size // y_1.size
        m = y_0.size % y_1.size

    repeat = np.repeat(file_less, n)
    choice = np.random.choice(file_less, m)
    files = np.hstack((files, repeat, choice))
    np.random.shuffle(files)

    y = np.array(get_labels(files), np.bool)[:, label_position]
    assert (y == 1).size == (y == 0).size
    return files


def get_labels(filenames):
    labels = []
    for i in filenames:
        label = i.split(".")[-2].split("_")[1:]
        labels.append(list(map(int, label)))
    return labels


def get_label(filename):
    label = filename.split(".")[-2].split("_")[1:]
    return label


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

    # 不对validation 数据集进行shuffle, 确保所有模型evaluate得出的结果是能够对应的，便于快速ensemble
    if shuffle:
        random.shuffle(train_files)

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


def load_label(directory, number=None):
    """
    导入指定目录的所有图片的标签，不导入图片
    :param directory:
    :return:
    """
    names = list_image_name(directory)
    random.shuffle(names)
    if number is not None:
        names = names[:number]
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


def image_repair():
    names = list_image_dir(path.ORIGINAL_TRAIN_IMAGES_PATH)
    for name in names:
        img = cv2.imread(name)
        cv2.imwrite(name, img)


if __name__ == '__main__':
    image_repair()
    # remove_image_name_header(path.ORIGINAL_TRAIN_IMAGES_PATH)
    # remove_image_name_header(path.SEGMENTED_TRAIN_IMAGES_PATH)

    # data_input_fn()
