import os
import pathlib

import config
import util

root_path = os.sep.join(util.__file__.split(os.sep)[:-2])
ROOT_PATH = os.path.abspath(root_path)
DATA_PATH = os.path.join(ROOT_PATH, "data")
MODEL_PATH = os.path.join(ROOT_PATH, "model")
ENSEMBLE_PATH = os.path.join(ROOT_PATH, "ensemble")

RESULT_PATH = os.path.join(DATA_PATH, "result")
CNN_RESULT_PATH = os.path.join(RESULT_PATH, "cnn")
XGB_RESULT_PATH = os.path.join(RESULT_PATH, "xgb")
SUBMIT_RESULT_PATH = os.path.join(RESULT_PATH, "submit")
pathlib.Path(CNN_RESULT_PATH).mkdir(parents=True, exist_ok=True)
pathlib.Path(XGB_RESULT_PATH).mkdir(parents=True, exist_ok=True)
pathlib.Path(SUBMIT_RESULT_PATH).mkdir(parents=True, exist_ok=True)

DATA_TYPE_ORIGINAL = "original"
DATA_TYPE_SEGMENTED = "segmented"

XGBOOST_ENSEMBLE_PATH = os.path.join(ENSEMBLE_PATH, "xgb")

CACHE_PATH = os.path.join(DATA_PATH, "cache")
IMAGES_PATH = os.path.join(DATA_PATH, "images")
SPIDER_PATH = os.path.join(DATA_PATH, 'spider')
K_FOLD_IMAGE_PATH = os.path.join(IMAGES_PATH, "k-fold")
ORIGINAL_IMAGES_PATH = os.path.join(IMAGES_PATH, config.DATA_TYPE_ORIGINAL)
SEGMENTED_IMAGES_PATH = os.path.join(IMAGES_PATH, config.DATA_TYPE_SEGMENTED)
AUGMENTED_IMAGES_PATH = os.path.join(IMAGES_PATH, "augmented")

COLOR_AUGMENTED_PATH = os.path.join(AUGMENTED_IMAGES_PATH, "color")
BLUE_IMAGES_PATH = os.path.join(COLOR_AUGMENTED_PATH, "blue")
BLUE_ORIGNINAL_PATH = os.path.join(BLUE_IMAGES_PATH, "original")
BLUE_TO_GREEN_PATH = os.path.join(BLUE_IMAGES_PATH, "green")
BLUE_TO_YELLOW_PATH = os.path.join(BLUE_IMAGES_PATH, "yellow")

GREEN_IMAGE_PATH = os.path.join(COLOR_AUGMENTED_PATH, "green")
GREEN_ORIGINAL_PATH = os.path.join(GREEN_IMAGE_PATH, "original")
GREEN_TO_YELLOW_PATH = os.path.join(GREEN_IMAGE_PATH, "yellow")

YELLOW_IMAGE_PATH = os.path.join(COLOR_AUGMENTED_PATH, "yellow")
YELLOW_ORIGINAL_PATH = os.path.join(YELLOW_IMAGE_PATH, "original")

ORANGE_IMAGE_PATH = os.path.join(COLOR_AUGMENTED_PATH, "orange")
ORANGE_ORIGINAL_PATH = os.path.join(ORANGE_IMAGE_PATH, "original")

RED_IMAGE_PATH = os.path.join(COLOR_AUGMENTED_PATH, "red")
RED_ORIGINAL_PATH = os.path.join(RED_IMAGE_PATH, "original")

RED_IMAGE_PATH = os.path.join(COLOR_AUGMENTED_PATH, "red")
RED_ORIGINAL_PATH = os.path.join(RED_IMAGE_PATH, "original")

DEBUG_TRAIN_IMAGES_PATH = os.path.join(os.path.join(IMAGES_PATH, "debug"), "train")
DEBUG_VAL_IMAGES_PATH = os.path.join(os.path.join(IMAGES_PATH, "debug"), "val")

ERROR_ORIGINAL_IMAGES_PATH = os.path.join(os.path.join(IMAGES_PATH, "error"), "original")
ERROR_IMAGE_PATH = os.path.join(IMAGES_PATH, "error")

TRAIN_IMAGES_SUBDIR = "train"
TEST_IMAGES_SUBDIR = "test"

ORIGINAL_TRAIN_IMAGES_PATH = os.path.join(ORIGINAL_IMAGES_PATH, TRAIN_IMAGES_SUBDIR)
ORIGINAL_TEST_IMAGES_PATH = os.path.join(ORIGINAL_IMAGES_PATH, TEST_IMAGES_SUBDIR)
SEGMENTED_TRAIN_IMAGES_PATH = os.path.join(SEGMENTED_IMAGES_PATH, TRAIN_IMAGES_SUBDIR)
SEGMENTED_TEST_IMAGES_PATH = os.path.join(SEGMENTED_IMAGES_PATH, TEST_IMAGES_SUBDIR)

TXT_PATH = os.path.join(DATA_PATH, "txt")
K_FOLD_TXT_PATH = os.path.join(TXT_PATH, "k-fold")
TRAIN_DATA_TXT = os.path.join(TXT_PATH, "train.txt")
TEST_DATA_TXT = os.path.join(TXT_PATH, "test.txt")
TEST_RESULT_TXT = os.path.join(TXT_PATH, "test-result.txt")
SUBMIT_DATA_TXT = os.path.join(TXT_PATH, "submit.txt")

IMAGE_STATISTICS_PATH = os.path.join(DATA_PATH, "statistics")
EPOCH_TEST_STANDARD = os.path.join(IMAGE_STATISTICS_PATH, "epoch_test_standard.json")
EPOCH_TEST = os.path.join(IMAGE_STATISTICS_PATH, "epoch_test.json")
EPOCH_CV = os.path.join(IMAGE_STATISTICS_PATH, "epoch_cv.json")
MODEL_TEST = os.path.join(IMAGE_STATISTICS_PATH, "model_test.json")
MODEL_CV = os.path.join(IMAGE_STATISTICS_PATH, "model_cv.json")
GLOBAL_TEST = os.path.join(IMAGE_STATISTICS_PATH, "global_test.json")
GLOBAL_CV = os.path.join(IMAGE_STATISTICS_PATH, "global_cv.json")
THRESHOLD_CV = os.path.join(IMAGE_STATISTICS_PATH, "threshold_cv.json")
THRESHOLD_TEST = os.path.join(IMAGE_STATISTICS_PATH, "threshold_test.json")



def get_image_mean_file(k_fold_file, val_index, data_type=(config.DATA_TYPE_SEGMENTED,)):
    pathlib.Path(os.path.join(os.path.join(IMAGE_STATISTICS_PATH, k_fold_file.split(".")[0]))) \
        .mkdir(parents=True, exist_ok=True)
    return os.path.join(os.path.join(IMAGE_STATISTICS_PATH, k_fold_file.split(".")[0]),
                        "val_%d_train_mean_%s.npy" % (val_index, str(data_type)))


def get_image_std_file(k_fold_file, val_index, rescale=256, data_type=(config.DATA_TYPE_SEGMENTED,)):
    pathlib.Path(os.path.join(os.path.join(IMAGE_STATISTICS_PATH, k_fold_file.split(".")[0]))) \
        .mkdir(parents=True, exist_ok=True)
    return os.path.join(os.path.join(IMAGE_STATISTICS_PATH, k_fold_file.split(".")[0]),
                        "val_%d_train_std_%s.npy" % (val_index, str(data_type)))


def get_train_data_path(data_type):
    return os.path.join(os.path.join(IMAGES_PATH, data_type), TRAIN_IMAGES_SUBDIR)


def get_test_data_path(data_type):
    return os.path.join(os.path.join(IMAGES_PATH, data_type), TEST_IMAGES_SUBDIR)


def image_path_init():
    pathlib.Path(SEGMENTED_TRAIN_IMAGES_PATH).mkdir(parents=True, exist_ok=True)
    pathlib.Path(ORIGINAL_TRAIN_IMAGES_PATH).mkdir(parents=True, exist_ok=True)
    pathlib.Path(ORIGINAL_TEST_IMAGES_PATH).mkdir(parents=True, exist_ok=True)

    pathlib.Path(os.path.join(K_FOLD_IMAGE_PATH, config.DATA_TYPE_ORIGINAL)).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(K_FOLD_IMAGE_PATH, config.DATA_TYPE_SEGMENTED)).mkdir(parents=True, exist_ok=True)

    pathlib.Path(BLUE_ORIGNINAL_PATH).mkdir(parents=True, exist_ok=True)
    pathlib.Path(BLUE_TO_GREEN_PATH).mkdir(parents=True, exist_ok=True)
    pathlib.Path(BLUE_TO_YELLOW_PATH).mkdir(parents=True, exist_ok=True)
    pathlib.Path(GREEN_ORIGINAL_PATH).mkdir(parents=True, exist_ok=True)
    pathlib.Path(GREEN_TO_YELLOW_PATH).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    image_path_init()
