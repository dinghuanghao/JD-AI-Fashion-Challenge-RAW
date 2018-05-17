import os
import pathlib
import shutil

import config
import util

root_path = "\\".join(util.__file__.split("\\")[:-2])
ROOT_PATH = os.path.abspath(root_path)
DATA_PATH = os.path.join(ROOT_PATH, "data")
TRAINING_RECORD_PATH = os.path.join(ROOT_PATH, "record")

IMAGES_PATH = os.path.join(DATA_PATH, "images")
K_FOLD_IMAGE_PATH = os.path.join(IMAGES_PATH, "k-fold")
ORIGINAL_IMAGES_PATH = os.path.join(IMAGES_PATH, config.DATA_TYPE_ORIGINAL)
SEGMENTED_IMAGES_PATH = os.path.join(IMAGES_PATH, config.DATA_TYPE_SEGMENTED)
AUGMENTED_IMAGES_PATH = os.path.join(IMAGES_PATH, config.DATA_TYPE_AUGMENTED)

TRAIN_IMAGES_SUBDIR = "train"
TEST_IMAGES_SUBDIR = "test"


ORIGINAL_TRAIN_IMAGES_PATH = os.path.join(ORIGINAL_IMAGES_PATH, TRAIN_IMAGES_SUBDIR)
ORIGINAL_TEST_IMAGES_PATH = os.path.join(ORIGINAL_IMAGES_PATH, TEST_IMAGES_SUBDIR)
SEGMENTED_TRAIN_IMAGES_PATH = os.path.join(SEGMENTED_IMAGES_PATH, TRAIN_IMAGES_SUBDIR)
SEGMENTED_TEST_IMAGES_PATH = os.path.join(SEGMENTED_IMAGES_PATH, TEST_IMAGES_SUBDIR)
AUGMENTED_TRAIN_IMAGES_PATH = os.path.join(AUGMENTED_IMAGES_PATH, TRAIN_IMAGES_SUBDIR)
AUGMENTED_TEST_IMAGES_PATH = os.path.join(AUGMENTED_IMAGES_PATH, TEST_IMAGES_SUBDIR)

TXT_PATH = os.path.join(DATA_PATH, "txt")
K_FOLD_TXT_PATH = os.path.join(TXT_PATH, "k-fold")
TRAIN_DATA_TXT = os.path.join(TXT_PATH, "train.txt")
TEST_DATA_TXT = os.path.join(TXT_PATH, "test.txt")
SUBMIT_DATA_TXT = os.path.join(TXT_PATH, "submit.txt")


def get_train_data_path(data_type):
    return os.path.join(os.path.join(IMAGES_PATH, data_type), TRAIN_IMAGES_SUBDIR)


def get_test_data_path(data_type):
    return os.path.join(os.path.join(IMAGES_PATH, data_type), TEST_IMAGES_SUBDIR)


def get_k_fold_data_path(data_type: str, batch: str, index: str = None):
    """
    获得K-FOLD DataSets例如：
    original, 1, 2：根据original数据和1.txt划分的第二份数据
    :param data_type: 如original，segmented，augmented
    :param batch: 1，2……，对应txt/k-fold/下的文件名
    :param index: 1~5，第几份数据
    :return:
    """
    if index is None:
        return os.path.join(K_FOLD_IMAGE_PATH, data_type, batch)
    return os.path.join(K_FOLD_IMAGE_PATH, data_type, batch, index)


def get_k_fold_txt_path(batch: str):
    """
    获取k-fold划分结果的文件路径（k-fold目录下的txt文件所在路径）
    :param batch:
    :return:
    """
    return os.path.join(K_FOLD_TXT_PATH, batch + ".txt")


def image_path_init():
    pathlib.Path(SEGMENTED_TRAIN_IMAGES_PATH).mkdir(parents=True, exist_ok=True)
    pathlib.Path(SEGMENTED_TEST_IMAGES_PATH).mkdir(parents=True, exist_ok=True)
    pathlib.Path(AUGMENTED_TRAIN_IMAGES_PATH).mkdir(parents=True, exist_ok=True)
    pathlib.Path(AUGMENTED_TEST_IMAGES_PATH).mkdir(parents=True, exist_ok=True)
    pathlib.Path(ORIGINAL_TRAIN_IMAGES_PATH).mkdir(parents=True, exist_ok=True)
    pathlib.Path(ORIGINAL_TEST_IMAGES_PATH).mkdir(parents=True, exist_ok=True)

    pathlib.Path(os.path.join(K_FOLD_IMAGE_PATH, config.DATA_TYPE_ORIGINAL)).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(K_FOLD_IMAGE_PATH, config.DATA_TYPE_SEGMENTED)).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(K_FOLD_IMAGE_PATH, config.DATA_TYPE_AUGMENTED)).mkdir(parents=True, exist_ok=True)


def refresh_k_fold_data(data_type: str, batch: str):
    if os.path.exists(get_k_fold_data_path(data_type, batch)):
        shutil.rmtree(get_k_fold_data_path(data_type, batch))
    os.mkdir(get_k_fold_data_path(data_type, batch))

    for i in range(config.K_FOLD):
        os.mkdir(get_k_fold_data_path(data_type, batch, str(i)))
        os.mkdir(get_k_fold_data_path(data_type, batch, str(i)))
        os.mkdir(get_k_fold_data_path(data_type, batch, str(i)))
        os.mkdir(get_k_fold_data_path(data_type, batch, str(i)))
        os.mkdir(get_k_fold_data_path(data_type, batch, str(i)))


if __name__ == "__main__":
    image_path_init()
