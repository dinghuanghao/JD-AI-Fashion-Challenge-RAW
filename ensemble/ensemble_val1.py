import os
import re
import config
from util import path
from util import keras_util
from util.keras_util import KerasModelConfig
from util import data_loader
import numpy as np

from ensemble import model_statistics

model_config = KerasModelConfig(k_fold_file="1.txt",
                                val_index=1,
                                model_path=os.path.abspath(__file__),
                                image_resolution=224,
                                data_type=[config.DATA_TYPE_SEGMENTED],
                                label_position=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                                train_batch_size=[32, 32, 32],
                                val_batch_size=512,
                                predict_batch_size=512,
                                epoch=[1, 6, 12],
                                lr=[0.001, 0.0001, 0.00001],
                                freeze_layers=[-1, 0.5, 0])

all_label, one_label = model_statistics.model_statistics(path.MODEL_PATH, val_index=model_config.val_index)

segment_train_files, segment_val_files = data_loader.get_k_fold_files(model_config.k_fold_file,
                                                                      model_config.val_index,
                                                                      [config.DATA_TYPE_SEGMENTED], shuffle=False)
y_true = np.array(data_loader.get_labels(segment_val_files), np.bool)[:, model_config.label_position]


# 需要val1~val5 均训练出多个模型，才能做集成




