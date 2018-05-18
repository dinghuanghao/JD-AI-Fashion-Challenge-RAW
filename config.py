import os
import pathlib
import time

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
PREFETCH_BUFFER_SIZE = 512


class ModelConfig(object):
    def __init__(self, k_fold_file, val_index, image_size, image_shape, data_type, model_dir, record_sub_dir, output_tensor_name, epoch = 10, batch_size=32) -> None:
        self.k_fold_file = k_fold_file
        self.val_index = val_index
        self.image_size = image_size
        self.image_shape = image_shape
        self.data_type = data_type
        self.model_dir = model_dir
        self.output_tensor_name = output_tensor_name
        self.epoch = epoch
        self.batch_size = batch_size
        self.record_dir = os.path.join(os.path.join(model_dir, "record"), str(record_sub_dir))
        pathlib.Path(self.record_dir).mkdir(parents=True, exist_ok=True)

    def save_before_train(self):
        log_file = os.path.join(self.record_dir, "log.txt")
        mode = "a"
        if not os.path.exists(log_file):
            mode = "w"

        with open(log_file, mode) as f:
            f.write("-------------------------------------------------------\n")
            f.write("-------------------------------------------------------\n")
            f.write("Start Time     : %s\n" % time.asctime(time.localtime(time.time())))
            f.write("K-fold file    : %s\n" % self.k_fold_file)
            f.write("Validation     : part%d\n" % self.val_index)
            f.write("ImageSize      : %s\n" % str(self.image_size))
            f.write("ImageShape     : %s\n" % str(self.image_shape))
            f.write("DataSet type   : %s\n" % str(self.data_type))
            f.write("Output Tensor  : %s\n" % self.output_tensor_name)
            f.write("Epoch          : %d\n" % self.epoch)
            f.write("Batch Size     : %d\n" % self.batch_size)
    def save_after_train(self):
        with open(os.path.join(self.record_dir, "log.txt"), "a") as f:
            f.write("Stop Time:     : %s\n" % time.asctime(time.localtime(time.time())))

