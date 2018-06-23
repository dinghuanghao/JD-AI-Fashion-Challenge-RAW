import concurrent.futures
import os
import shutil
import time

import cv2
import numpy as np
from tqdm import tqdm

from util import data_loader
from util import data_metric
from util import path

threshold = [640000 / 15, 640000 / 2]

lower_blue = np.array([90, 50, 80])
upper_blue = np.array([120, 255, 255])

lower_green = np.array([50, 50, 80])
upper_green = np.array([75, 255, 255])

lower_pink = np.array([165, 20, 160])
upper_pink = np.array([180, 100, 255])

lower_red = np.array([175, 178, 50])
upper_red = np.array([360, 255, 255])

lower_yellow = np.array([24, 128, 200])
upper_yellow = np.array([30, 255, 255])

lower_orange = np.array([13, 128, 50])
upper_orange = np.array([23, 255, 255])


def copy_images_by_hsv(image_paths, target_path, lower, upper, thread_id):
    print("thread %d start" % thread_id)

    if thread_id == 0:
        for image_path in tqdm(image_paths):
            image_name = os.path.split(image_path)[-1]
            if os.path.exists(os.path.join(target_path, image_name)):
                continue

            img = cv2.imread(image_path)
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(img_hsv, lower, upper)

            mask_size = mask[mask > 200].size
            if threshold[1] > mask_size > threshold[0]:
                original_file = os.path.join(path.ORIGINAL_TRAIN_IMAGES_PATH, image_name)
                shutil.copyfile(original_file, os.path.join(target_path, image_name))
    else:
        for image_path in image_paths:
            image_name = os.path.split(image_path)[-1]
            if os.path.exists(os.path.join(target_path, image_name)):
                continue

            img = cv2.imread(image_path)
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(img_hsv, lower, upper)

            mask_size = mask[mask > 200].size
            if threshold[1] > mask_size > threshold[0]:
                original_file = os.path.join(path.ORIGINAL_TRAIN_IMAGES_PATH, image_name)
                shutil.copyfile(original_file, os.path.join(target_path, image_name))


def copy_images_multi_thread(image_paths, target_path, lower, upper, thread_number=4):
    lines = []
    for i in range(thread_number):
        lines.append([])
    line_num = 0
    for l in image_paths:
        lines[line_num % thread_number].append(l)
        line_num += 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=thread_number) as executor:
        futures = []
        start_time = time.time()
        for i in range(len(lines)):
            futures.append(executor.submit(copy_images_by_hsv, lines[i], target_path, lower, upper, i))
        concurrent.futures.wait(futures)
        print("拷贝所有图片共花费 %f 秒" % (time.time() - start_time))


def get_augment_image_dirs():
    images = data_loader.list_image_dir(path.COLOR_AUGMENTED_PATH)
    augment_images = []
    for image in images:
        if "original" not in image:
            augment_images.append(image)
    return augment_images

def do_statistic():
    images = data_loader.list_image_dir(path.COLOR_AUGMENTED_PATH)
    augment_images = []
    for image in images:
        if "original" not in image:
            augment_images.append(image)

    labels = data_loader.get_labels(augment_images)
    labels = np.array(labels)
    data_metric.label_analysis(labels)


if __name__ == "__main__":
    do_statistic()
    # images = data_loader.list_image_dir(path.SEGMENTED_TRAIN_IMAGES_PATH)
    # copy_images_multi_thread(images, path.ORANGE_ORIGINAL_PATH, lower_orange, upper_orange, 12)
