DEMO_TRAINING_TXT_PATH = "../data/demo/txt/sample-train.txt"
DEMO_TEST_TXT_PATH = "../data/demo/txt/sample-test.txt"

DEMO_PHOTOS_PATH = "../data/demo/photos"
TEST_SUBDIR = "test"
TRAINING_SUBDIR = "train"

import os

import keras.utils as kutil

print(os.listdir("../"))


def parse_data_line(line: str):
    pieces = line.strip().split(",")
    return pieces[0], pieces[1], "".join(pieces[2:])


# TODO: Download images via multithreading
def download_photos(txt_dir: str, photo_save_dir: str, photo_save_subdir: str, is_test: bool):
    with open(txt_dir) as f:
        lines = f.readlines()

        for l in f.readlines():
            name, url, label = parse_data_line(l)
            if is_test:
                kutil.get_file(fname=name+".jpg", origin=url, cache_dir=photo_save_dir, cache_subdir=photo_save_subdir)
            else:
                kutil.get_file(fname=label+".jpg", origin=url, cache_dir=photo_save_dir, cache_subdir=photo_save_subdir)


if __name__ == '__main__':
    download_photos(DEMO_TRAINING_TXT_PATH, DEMO_PHOTOS_PATH, TRAINING_SUBDIR)
    download_photos(DEMO_TEST_TXT_PATH, DEMO_PHOTOS_PATH, TEST_SUBDIR)
