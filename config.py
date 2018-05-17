import os

# 图片尺寸
IMAGE_SIZE = [224, 224]
IMAGE_SHAPE = (224, 224, 3)

# K_FOLD次数
K_FOLD = 5

# 数据集的类型，用于命名文件夹和图片
DATA_TYPE_ORIGINAL = "original"
DATA_TYPE_SEGMENTED = "segmented"
DATA_TYPE_AUGMENTED = "augmented"

# 用于图片的下载，文件需要放到data/txt目录下
TRAIN_DATA_TXT = "train.txt"
TEST_DATA_TXT = "test.txt"

EPOCH = 10
BATCH_SIZE = 32
PREFETCH_BUFFER_SIZE = 32


class ModelConfig(object):
    def __init__(self, k_fold_file, val_index, image_size, image_shape, data_type, model_dir) -> None:
        self.k_fold_file = k_fold_file
        self.val_index = val_index
        self.image_size = image_size
        self.image_shape = image_shape
        self.data_type = data_type
        self.model_dir = model_dir

