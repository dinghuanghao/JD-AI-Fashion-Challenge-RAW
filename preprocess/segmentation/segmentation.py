import os

import numpy as np
import skimage.io
import config

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
        IMAGES_PER_GPU = 1

    config = InferenceConfig()
    config.display()

    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    # Load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    return model


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

    image_paths = [os.path.join(image_dir, name) for name in image_names]
    image_save_paths = [os.path.join(save_dir, "_".join([config.DATA_TYPE_SEGMENTED, name])) for name in image_names]

    for i in range(len(image_names)):
        image = skimage.io.imread(image_paths[i])
        results = model.detect([image], verbose=1)

        r = results[0]

        if show_image:
            visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], CLASS_NAME, class_used,
                                        r['scores'])

        image_masked = image_masking(image, r['rois'], r['masks'], r['class_ids'], CLASS_NAME, class_used)

        if show_image:
            visualize.display_image(image_masked)

        skimage.io.imsave(image_save_paths[i], image_masked)


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
    mask = 0
    for i in range(N):
        if class_names[class_ids[i]] not in class_used:
            continue
        if person_num == 0:
            person_num += 1
            mask = masks[:, :, i]
        else:
            mask = masks[:, :, i] | mask

    if person_num != 0 and mask.shape[0] > 0:
        for c in range(3):
            masked_image[:, :, c] = np.where(mask == 1, masked_image[:, :, c], white[:, :, c])

    return masked_image


if __name__ == "__main__":
    model = get_mrcnn_model()
    image_names = data_loader.list_image_name(path.ORIGINAL_TEST_IMAGES_PATH)

    # 测试单张图片
    # image_segmentation(model, path.ORIGINAL_TEST_IMAGES_PATH, ["582f0f9eNbc1d2a40.jpg"],path.SEGMENTED_TEST_IMAGES_PATH, True)

    # 将原始数据进行人像分割并保存
    image_segmentation(model, path.ORIGINAL_TEST_IMAGES_PATH, image_names, path.SEGMENTED_TEST_IMAGES_PATH, ['person'], False)
