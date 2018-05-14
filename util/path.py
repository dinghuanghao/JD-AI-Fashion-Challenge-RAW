import os

import util

root_path = "\\".join(util.__file__.split("\\")[:-2])
ROOT_PATH = os.path.abspath(root_path)
DATA_PATH = os.path.join(ROOT_PATH, "data")
TXT_PATH = os.path.join(DATA_PATH, "txt")
IMAGES_PATH = os.path.join(DATA_PATH, "images")
ORIGINAL_IMAGES_PATH = os.path.join(IMAGES_PATH, "original")
SEGMENTED_IMAGES_PATH = os.path.join(IMAGES_PATH, "segmented")
K_FOLD_PATH = os.path.join(IMAGES_PATH, "k-fold")

TRAIN_IMAGES_SUBDIR = "train"
TEST_IMAGES_SUBDIR = "test"
ORIGINAL_TRAIN_IMAGES_PATH = os.path.join(ORIGINAL_IMAGES_PATH, TRAIN_IMAGES_SUBDIR)
ORIGINAL_TEST_IMAGES_PATH = os.path.join(ORIGINAL_IMAGES_PATH, TEST_IMAGES_SUBDIR)
SEGMENTED_TRAIN_IMAGES_PATH = os.path.join(SEGMENTED_IMAGES_PATH, TRAIN_IMAGES_SUBDIR)
SEGMENTED_TEST_IMAGES_PATH = os.path.join(SEGMENTED_IMAGES_PATH, TEST_IMAGES_SUBDIR)

K_FOLD_ORIGINAL_SUBDIR = "%s-original"
K_FOLD_SEGMENTED_SUBDIR = "%s-sigmented"
K_FOLD_AUGMENTED_SUBDIR = "%s-augmented"

K_FOLD_1_ORIGINAL_PATH = os.path.join(K_FOLD_PATH, "1-original")
K_FOLD_2_ORIGINAL_PATH = os.path.join(K_FOLD_PATH, "2-original")
K_FOLD_3_ORIGINAL_PATH = os.path.join(K_FOLD_PATH, "3-original")
K_FOLD_4_ORIGINAL_PATH = os.path.join(K_FOLD_PATH, "4-original")
K_FOLD_5_ORIGINAL_PATH = os.path.join(K_FOLD_PATH, "5-original")

K_FOLD_1_SEGMENTED_PATH = os.path.join(K_FOLD_PATH, "1-segmented")
K_FOLD_2_SEGMENTED_PATH = os.path.join(K_FOLD_PATH, "2-segmented")
K_FOLD_3_SEGMENTED_PATH = os.path.join(K_FOLD_PATH, "3-segmented")
K_FOLD_4_SEGMENTED_PATH = os.path.join(K_FOLD_PATH, "4-segmented")
K_FOLD_5_SEGMENTED_PATH = os.path.join(K_FOLD_PATH, "5-segmented")

TRAIN_DATA_TXT = os.path.join(TXT_PATH, "train.txt")
TEST_DATA_TXT = os.path.join(TXT_PATH, "test.txt")
SUBMIT_DATA_TXT = os.path.join(TXT_PATH, "submit.txt")
K_FOLD_DATA_TXT = os.path.join(TXT_PATH, "k-fold.txt")
