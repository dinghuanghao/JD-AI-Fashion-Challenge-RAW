
#图片尺寸
IMAGE_SIZE = (224, 224)
IMAGE_SHAPE = (224, 224, 3)

#K_FOLD次数
K_FOLD = 5

#数据集的类型，用于命名文件夹和图片
DATA_TYPE_ORIGINAL = "original"
DATA_TYPE_SEGMENTED = "segmented"
DATA_TYPE_AUGMENTED = "augmented"

#用于图片的下载，文件需要放到data/txt目录下
TRAIN_DATA_TXT = "train.txt"
TEST_DATA_TXT = "test.txt"