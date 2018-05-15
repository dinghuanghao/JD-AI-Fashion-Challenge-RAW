import os
from shutil import copyfile

from util import path
import config


def copy_k_fold_images(data_type:str, batch:str):
    """
    根据k-fold.txt文件的内容，将文件从original目录拷贝到k-fold/1~5-original目录下
    :return:
    """
    path.refresh_k_fold_data(data_type, batch)
    with open(path.get_k_fold_txt_path(batch), "r") as f:
        for l in f.readlines():
            number, image_name = l.split(",")
            src_image_path = os.path.join(path.ORIGINAL_TRAIN_IMAGES_PATH, image_name).strip()
            dst_image_path = os.path.join(os.path.join(path.get_k_fold_data_path("original", batch, str(number)),image_name)).strip()
            copyfile(src_image_path, dst_image_path)




if __name__ == "__main__":
    copy_k_fold_images(config.DATA_TYPE_ORIGINAL, 1)
