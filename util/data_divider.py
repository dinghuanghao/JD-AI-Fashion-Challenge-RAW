import os
from shutil import copyfile

from  util import path


def copy_original_k_fold():
    """
    根据k-fold.txt文件的内容，将文件从original目录拷贝到k-fold/1~5-original目录下
    :return:
    """
    with open(path.K_FOLD_DATA_TXT, "r") as f:
        for l in f.readlines():
            number, image_name = l.split(",")
            src_image_path = os.path.join(path.ORIGINAL_TRAIN_IMAGES_PATH, image_name).strip()
            dst_image_path = os.path.join(os.path.join(path.K_FOLD_PATH, path.K_FOLD_ORIGINAL_SUBDIR % number),
                                          image_name).strip()
            copyfile(src_image_path, dst_image_path)


if __name__ == "__main__":
    copy_original_k_fold()
