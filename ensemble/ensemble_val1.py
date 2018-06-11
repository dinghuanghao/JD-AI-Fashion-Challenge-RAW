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
original_train_files, original_val_files = data_loader.get_k_fold_files(model_config.k_fold_file,
                                                                        model_config.val_index,
                                                                        [config.DATA_TYPE_ORIGINAL], shuffle=False)

y_valid_segment = np.array(data_loader.get_labels(segment_val_files), np.bool)[:, model_config.label_position]
y_valid_original = np.array(data_loader.get_labels(original_val_files), np.bool)[:, model_config.label_position]

val_file = {
    config.DATA_TYPE_ORIGINAL: segment_val_files,
    config.DATA_TYPE_SEGMENTED: original_val_files
}

assert (y_valid_original == y_valid_segment).all()

y_pred = None
weight_files = []
predictions = {}
for label, value in one_label.items():
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
            model = attr_get_model(output_dim=len(attr_model_config.label_position), weights=None)
            model.load_weights(weight_file)

            # 预测该模型的所有类型，并取平均
            prediction = None
            for data_type in attr_model_config.data_type:
                print("predict %s data for model %s" % (data_type, attr_model_config.model_path))
                if prediction is None:
                    prediction = keras_util.predict(model, val_file[data_type], attr_model_config, verbose=1)
                else:
                    prediction += keras_util.predict(model, val_file[data_type], attr_model_config, verbose=1)

            prediction = prediction / len(attr_model_config.data_type)
            predictions[weight_file] = prediction.copy()

        keras_util.evaluate(y_valid_segment, y_pred, "\n" + "\n".join(weight_files), model_config)
        for l in range(prediction.shape[-1]):
            if l != label:
                prediction[:, l] = 0

        y_pred = prediction if y_pred is None else y_pred + prediction
        keras_util.evaluate(y_valid_segment, y_pred, "\n" + "\n".join(weight_files), model_config)

        print("predict %s label" % label)
        break

keras_util.evaluate(y_valid_segment, y_pred, "\n" + "\n".join(weight_files), model_config)
