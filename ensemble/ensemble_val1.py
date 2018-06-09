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
                                train_batch_size=32,
                                val_batch_size=512,
                                predict_batch_size=512,
                                epoch=[1, 6, 12],
                                lr=[0.001, 0.0001, 0.00001],
                                freeze_layers=[-1, 0.5, 0])

all_label, one_label = model_statistics.model_statistics(path.MODEL_PATH, val_index=model_config.val_index)

# model_config.val_files = model_config.val_files[:64]
y_valid = np.array(data_loader.get_labels(model_config.val_files), np.bool)[:, model_config.label_position]

y_pred = None
weight_files = []
predictions = {}
for key, value in one_label.items():
    for i in value:
        weight_file, f2_score = i
        weight_files.append(weight_file)
        if weight_file in predictions:
            prediction = predictions[weight_file].copy()
        else:
            model_file = "_".join(re.match(r".*record\\(.*)\\\[", weight_file).group(1).split("\\"))
            model_dir = re.match(r"(.*)\\record", weight_file).group(1)
            model_path = os.path.join(model_dir, model_file)
            model_root_dir, model_type_dir, model_name = re.match(r".*competition\\(.*)", model_path).group(1).split(
                "\\")
            package = __import__(".".join([model_root_dir, model_type_dir, model_name]))
            attr_get_model = getattr(getattr(getattr(package, model_type_dir), model_name), "get_model")
            attr_model_config = getattr(getattr(getattr(package, model_type_dir), model_name), "model_config")
            model = attr_get_model(output_dim=len(attr_model_config.label_position))
            model.load_weights(weight_file)
            prediction = keras_util.predict(model, model_config.val_files, attr_model_config, verbose=1)
            predictions[weight_file] = prediction.copy()

        keras_util.evaluate(y_valid, prediction, weight_file, model_config)

        for j in range(prediction.shape[-1]):
            if j != key:
                prediction[:, j] = 0

        y_pred = prediction if y_pred is None else y_pred + prediction

        keras_util.evaluate(y_valid, y_pred, "\n" + "\n".join(weight_files), model_config)
        print("predict %s label" % key)
        break

keras_util.evaluate(y_valid, y_pred, "\n" + "\n".join(weight_files), model_config)
