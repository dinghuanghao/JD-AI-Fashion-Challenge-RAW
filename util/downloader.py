import concurrent.futures
import time

import keras.utils as kutil

DEMO_TRAINING_TXT_PATH = "../data/demo/txt/sample-train.txt"
DEMO_TEST_TXT_PATH = "../data/demo/txt/sample-test.txt"

DEMO_PHOTOS_PATH = "../data/demo/photos"
TEST_SUBDIR = "test"
TRAINING_SUBDIR = "train"


def parse_data_line(line: str):
    pieces = line.strip().split(",")
    return pieces[0], pieces[1], "".join(pieces[2:])


def do_download(names: list(), photo_save_dir: str, photo_save_subdir: str, is_test: bool):
    print("开始下载图片")
    for n in names:
        name, url, label = parse_data_line(n)
        if is_test:
            kutil.get_file(fname=name + ".jpg", origin=url, cache_dir=photo_save_dir,
                           cache_subdir=photo_save_subdir)
        else:
            kutil.get_file(fname=name + "_" + label + ".jpg", origin=url, cache_dir=photo_save_dir,
                           cache_subdir=photo_save_subdir)


def download_photos(txt_dir: str, photo_save_dir: str, photo_save_subdir: str, is_test: bool, thread_number: int = 2):
    """ 使用多线程下载可以成倍的提高下载速度，但是线程数过多后，服务器端会拒绝访问，本地测试结果是最多开2个线程 """
    with open(txt_dir) as f:
        lst = []
        for i in range(thread_number):
            lst.append([])
        line_num = 0
        for l in f.readlines():
            lst[line_num % thread_number].append(l)
            line_num += 1
        with concurrent.futures.ThreadPoolExecutor(max_workers=thread_number) as executor:
            futures = []
            start_time = time.time()
            for l in lst:
                futures.append(executor.submit(do_download(l, photo_save_dir, photo_save_subdir, is_test)))
            concurrent.futures.wait(futures)
            print("下载所有图片共花费 %f 秒" % (time.time() - start_time))


if __name__ == '__main__':
    download_photos(DEMO_TRAINING_TXT_PATH, DEMO_PHOTOS_PATH, TRAINING_SUBDIR, is_test=False, thread_number=2)
    download_photos(DEMO_TEST_TXT_PATH, DEMO_PHOTOS_PATH, TEST_SUBDIR, is_test=True, thread_number=2)
