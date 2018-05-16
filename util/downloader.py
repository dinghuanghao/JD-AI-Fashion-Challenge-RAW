
import concurrent.futures
import time

import tensorflow as tf
from util import path
import config


def parse_data_line(line: str):
    """
    处理图片官方提供的数据，得到图片编号，url，类型

    :param line: 待处理的数据
    :return: 图片编号、URL、类型
    """

    pieces = line.strip().split(",")
    return pieces[0], pieces[1], "_".join(pieces[2:])


def do_download(lines: list(), photo_save_dir: str, photo_save_subdir: str, is_test: bool):
    """
    下载图片

    :param lines: 待下载的数据
    :param photo_save_dir: 存储下载图片的根目录
    :param photo_save_subdir: 存储下载图片的子目录，根目录+子目录构成完整目录
    :param is_test: txt_dir对应的文本是否是测试数据（测试数据和训练数据的格式不同）
    :return:
    """

    print("开始下载图片")
    for n in lines:
        name, url, label = parse_data_line(n)
        while True:
            download_ok = True
            try:
                if is_test:
                    tf.keras.utils.get_file(fname=name + ".jpg", origin=url, cache_dir=photo_save_dir,
                                            cache_subdir=photo_save_subdir)
                else:
                    tf.keras.utils.get_file(fname="_".join([name, label]) + ".jpg", origin=url, cache_dir=photo_save_dir,
                                            cache_subdir=photo_save_subdir)
            except Exception as e:
                download_ok = False
                print(e)
                print("start retry")
            if download_ok:
                break



def download_photos(txt_dir: str, photo_save_dir: str, photo_save_subdir: str, is_test: bool, thread_number: int = 2):
    """ 使用多线程下载图片到制定的目录，线程过多可能导致服务器拒绝，本次测试2个线程比较稳定
        如果中途失败直接重试即可，不会重复下载已下载的图片

    :param txt_dir: 记录数据的文本的路径
    :param photo_save_dir: 存储下载图片的根目录
    :param photo_save_subdir: 存储下载图片的子目录，根目录+子目录构成完整目录
    :param is_test: txt_dir对应的文本是否是测试数据（测试数据和训练数据的格式不同）
    :param thread_number: 使用多线程下载的线程数量
    :return:
    """

    with open(txt_dir) as f:
        lines = []
        for i in range(thread_number):
            lines.append([])
        line_num = 0
        for l in f.readlines():
            lines[line_num % thread_number].append(l)
            line_num += 1
        with concurrent.futures.ThreadPoolExecutor(max_workers=thread_number) as executor:
            futures = []
            start_time = time.time()
            for l in lines:
                futures.append(executor.submit(do_download(l, photo_save_dir, photo_save_subdir, is_test)))
            concurrent.futures.wait(futures)
            print("下载所有图片共花费 %f 秒" % (time.time() - start_time))


if __name__ == '__main__':
    download_photos(path.TRAIN_DATA_TXT, path.ORIGINAL_IMAGES_PATH, path.TRAIN_IMAGES_SUBDIR, is_test=False, thread_number=16)
    #download_photos(path.TEST_DATA_TXT, path.ORIGINAL_IMAGES_PATH, path.TEST_IMAGES_SUBDIR, is_test=True, thread_number=1)
