import os
import pathlib

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


if __name__ == "__main__":
    image_path_init()
