import os
import time
import threading
import queue

import numpy as np
import skimage.io
import config
import cv2

import preprocess.segmentation.mrcnn.model as modellib
from preprocess.segmentation import coco  # Import COCO config
from preprocess.segmentation.mrcnn import utils  # Import Mask RCNN
from preprocess.segmentation.mrcnn import visualize
from util import data_loader
from util import path

ROOT_DIR = os.path.abspath(".")  # Root directory of the project
MODEL_DIR = os.path.join(ROOT_DIR, "logs")  # Directory to save logs and trained model
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# COCO 数据集中的类别，表示可进行识别的类
CLASS_NAME = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
              'bus', 'train', 'truck', 'boat', 'traffic light',
              'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
              'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
              'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
              'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
              'kite', 'baseball bat', 'baseball glove', 'skateboard',
              'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
              'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
              'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
              'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
              'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
              'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
              'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
              'teddy bear', 'hair drier', 'toothbrush']


# 仅仅对person进行分割



def get_mrcnn_model():
    """
    初始化MRCNN模型，使用预训练权重
    :return:
    """
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)

    class InferenceConfig(coco.CocoConfig):
        # 目前仅有一块1050ti在跑
        GPU_COUNT = 1
        IMAGES_PER_GPU = 10

    config = InferenceConfig()
    config.display()

    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    # Load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    return model


class ImageSaver(threading.Thread):
    def __init__(self, class_used, show_image, q: queue.Queue):
        threading.Thread.__init__(self)
        self.q = q
        self.class_used = class_used
        self.show_image = show_image

    def run(self):
        print("start image saver thread")
        while True:
            results, image_batch, image_save_path_batch = self.q.get()
            print("get %d results" % len(results))
            for index in range(len(results)):
                r = results[index]
                if self.show_image:
                    visualize.display_instances(image_batch[index], r['rois'], r['masks'], r['class_ids'],
                                                CLASS_NAME, self.class_used,
                                                r['scores'])

                image_masked = image_masking(image_batch[index], r['rois'], r['masks'], r['class_ids'], CLASS_NAME,
                                             self.class_used)

                if self.show_image:
                    visualize.display_image(image_masked)

                cv2.imwrite(image_save_path_batch[index], image_masked)
            self.q.task_done()


class ImageLoader(threading.Thread):
    def __init__(self, image_paths, image_save_paths, q: queue.Queue):
        threading.Thread.__init__(self)
        self.q = q
        self.image_paths = image_paths
        self.image_save_paths = image_save_paths

    def run(self):
        print("start image loader thread")
        mode = "a"
        if not os.path.exists("segment.log"):
            mode = "w"

        image_number = len(image_names)
        image_batch = []
        image_save_path_batch = []

        for i in range(image_number):
            if os.path.exists(self.image_save_paths[i]):
                continue
            try:
                image = cv2.imread(self.image_paths[i])
                image_batch.append(image)
                image_save_path_batch.append(self.image_save_paths[i])
                if len(image_batch) == model.config.IMAGES_PER_GPU:
                    self.q.put((image_batch, image_save_path_batch))
                    print("put item to queue, queue len=%d" % self.q.qsize())
                    image_batch = []
                    image_save_path_batch = []

            except Exception as e:
                with open("segment.log", mode) as f:
                    f.write("%s(%s): %s\n" % (time.asctime(time.localtime(time.time())), image_names[i], e))

        if len(image_batch) != 0:
            print("put last item to queue, queue len=%d" % self.q.qsize())
            self.q.put((image_batch, image_save_path_batch))


def image_segmentation(model, image_dir, image_names: [], save_dir, class_used=("person"), show_image=False):
    """
    对图像进行分割并保存
    :param model: MRCNN模型
    :param image_dir: 图像所在路径
    :param image_names: 图像名称列表，用于图像的读取和存储
    :param save_dir: 图像保存路径
    :param class_used: 需要分割的类别
    :param show_image: 是否要对分割结果进行展示
    :return:
    """

    mode = "a"
    if not os.path.exists("segment.log"):
        mode = "w"

    image_paths = [os.path.join(image_dir, name) for name in image_names]
    image_save_paths = [os.path.join(save_dir, name) for name in image_names]

    image_queue = queue.Queue(maxsize=10)
    result_queue = queue.Queue()

    image_loader = ImageLoader(image_paths, image_save_paths, image_queue)
    image_loader.setDaemon(True)
    image_loader.start()

    image_saver = ImageSaver(class_used, show_image, result_queue)
    image_saver.setDaemon(True)
    image_saver.start()

    while True:
        try:
            image_batch, image_save_path_batch = image_queue.get()
            if len(image_batch) < model.config.IMAGES_PER_GPU:
                class InferenceConfig(coco.CocoConfig):
                    # 目前仅有一块1050ti在跑
                    GPU_COUNT = 1
                    IMAGES_PER_GPU = len(image_batch)

                config = InferenceConfig()

                # 此处之所以要重新建一个模型，是因为之前的模型已经发送到了GPU上，单纯在CPU上修改config没有作用
                model_1 = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
                model.load_weights(COCO_MODEL_PATH, by_name=True)
                results = model_1.detect(image_batch, verbose=0)
                result_queue.put((results, image_batch, image_save_path_batch))
                break
            else:
                results = model.detect(image_batch, verbose=0)
                result_queue.put((results, image_batch, image_save_path_batch))
        except Exception as e:
            with open("segment.log", mode) as f:
                f.write("%s(%s): %s\n" % (time.asctime(time.localtime(time.time())), image_names[i], e))

    result_queue.join()


def image_masking(image, boxes, masks, class_ids, class_names, class_used):
    """
    将掩码应用到图像上，把不相关的地方置为白色
    :param image:
    :param boxes:
    :param masks:
    :param class_ids:
    :param class_names:
    :param class_used:
    :return:
    """

    masked_image = image
    white = np.ones(image.shape) * 255
    N = boxes.shape[0]

    person_num = 0
    max_mask = None
    max_mask_size = 0
    for i in range(N):
        if class_names[class_ids[i]] not in class_used:
            continue

        person_num += 1
        if masks[:, :, i].size > max_mask_size:
            max_mask_size = masks[:, :, i].size
            max_mask = masks[:, :, i]

    if person_num != 0 and max_mask.shape[0] > 0:
        for c in range(3):
            masked_image[:, :, c] = np.where(max_mask == 1, masked_image[:, :, c], white[:, :, c])

    return masked_image


if __name__ == "__main__":
    model = get_mrcnn_model()
    # image_names = data_loader.list_image_name(path.ORIGINAL_TRAIN_IMAGES_PATH)
    #
    # # 将原始数据进行人像分割并保存
    # image_segmentation(model, path.ORIGINAL_TRAIN_IMAGES_PATH, image_names, path.SEGMENTED_TRAIN_IMAGES_PATH,
    #                    ['person'], False)
    image_names = data_loader.list_image_name(path.ERROR_ORIGINAL_IMAGES_PATH)

    # 将原始数据进行人像分割并保存
    start = time.time()
    image_segmentation(model, path.ERROR_ORIGINAL_IMAGES_PATH, image_names, path.ERROR_IMAGE_PATH, ['person'], False)
    print("#############spend %d seconds" % (time.time() - start))
