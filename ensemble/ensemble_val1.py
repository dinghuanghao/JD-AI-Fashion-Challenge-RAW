import os
import re
import config
from util import path
from util import keras_util
from util.keras_util import KerasModelConfig

import keras
from ensemble import model_statistics

model_config = KerasModelConfig(k_fold_file="1.txt",
                                val_index=1,
                                model_path=os.path.abspath(__file__),
                                image_resolution=224,
                                data_type=[config.DATA_TYPE_SEGMENTED],
                                label_position=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                                train_batch_size=32,
                                val_batch_size=64,
                                predict_batch_size=128,
                                epoch=[1, 6, 12],
                                lr=[0.001, 0.0001, 0.00001],
                                freeze_layers=[-1, 0.5, 0])

#应该基于val进行统计
all_label, one_label = model_statistics.model_statistics(path.MODEL_PATH)

model_config.val_files = model_config.val_files[:128]

for key, value in one_label.items():
    for i in value:
        weight_file, f2_score = i
        model_file = "_".join(re.match(r".*record\\(.*)\\\[", weight_file).group(1).split("\\"))
        model_dir = re.match(r"(.*)\\record", weight_file).group(1)
        model_path = os.path.join(model_dir, model_file)
        model_root_dir, model_type_dir, model_name = re.match(r".*competition\\(.*)", model_path).group(1).split("\\")
        package = __import__(".".join([model_root_dir, model_type_dir, model_name]))
        attr_get_model = getattr(getattr(getattr(package, model_type_dir), model_name), "get_model")
        attr_model_config = getattr(getattr(getattr(package, model_type_dir), model_name), "model_config")
        model = attr_get_model(output_dim=len(attr_model_config.label_position))
        model.load_weights(weight_file)
        y_pred = keras_util.predict(model, model_config.val_files, attr_model_config)

        for j in range(len(attr_model_config.label_position)):
            if j != key:
                y_pred[:, j] = 0

        print("predict %s label" % key)
        break

