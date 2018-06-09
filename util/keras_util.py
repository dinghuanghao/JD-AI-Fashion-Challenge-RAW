import os
import pathlib
import time
import math

import keras
import numpy as np

from util import data_loader
from util import metrics
from util import path


class KerasModelConfig(object):
    def __init__(self,
                 k_fold_file,
                 model_path:str,
                 image_resolution,
                 data_type,
                 label_position=(1,),
                 train_batch_size=32,
                 val_batch_size=32,
                 predict_batch_size=32,
                 epoch=(1,),
                 lr=(0.01,),
                 freeze_layers=(0,)):

        file_name = os.path.basename(model_path)
        model_dir = os.path.dirname(model_path)

        self.k_fold_file = k_fold_file
        self.val_index = int("".join(filter(str.isdigit, file_name.split("_")[1])))
        self.image_resolution = image_resolution
        self.image_size = (image_resolution, image_resolution)
        self.image_shape = (image_resolution, image_resolution, 3)
        self.data_type = data_type
        self.record_dir = os.path.join(os.path.join(model_dir, "record"), file_name.split("_")[0])
        self.record_dir = os.path.join(self.record_dir, "val%d" % self.val_index)
        self.label_position = label_position
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.predict_batch_size = predict_batch_size
        self.epoch = epoch
        self.lr = lr
        self.freeze_layers = freeze_layers
        self.train_files, self.val_files = data_loader.get_k_fold_files(self.k_fold_file, self.val_index,
                                                                        self.data_type)
        self.train_files = np.array(self.train_files)
        self.val_files = np.array(self.val_files)
        self.image_mean_file = path.get_image_mean_file(self.k_fold_file, self.val_index,
                                                        data_type=self.data_type)
        self.image_std_file = path.get_image_std_file(self.k_fold_file, self.val_index,
                                                      data_type=self.data_type)

        self.save_model_format = os.path.join(self.record_dir,
                                              "%sweights.{epoch:03d}.hdf5" % str([str(i) for i in self.label_position]))
        self.tem_model_file = os.path.join(self.record_dir, 'weights.hdf5')
        pathlib.Path(self.record_dir).mkdir(parents=True, exist_ok=True)

        print("file name is: %s" % file_name)
        print("val index is: %d" % self.val_index)
        print("model dir is: %s" % model_dir)
        print("record dir is: %s" % self.record_dir)

    def get_steps_per_epoch(self):
        return math.ceil(len(self.train_files) / self.train_batch_size)

    def get_weights_path(self, epoch):
        if epoch < 10:
            return os.path.join(self.record_dir, "%sweights.00%d.hdf5" % (str([str(j) for j in self.label_position]), epoch))
        else:
            return os.path.join(self.record_dir,
                                "%sweights.0%d.hdf5" % (str([str(j) for j in self.label_position]), epoch))


def evaluate(model: keras.Model, pre_files, y, weight_name, model_config: KerasModelConfig):
    from sklearn.metrics import fbeta_score

    pre_datagen = data_loader.KerasGenerator(featurewise_center=True,
                                             featurewise_std_normalization=True,
                                             rescale=1. / 256)

    data_loader.check_mean_std_file(model_config, pre_datagen)
    pre_datagen.load_image_global_mean_std(model_config.image_mean_file, model_config.image_std_file)

    pre_flow = pre_datagen.flow_from_files(pre_files, mode="predict",
                                           target_size=model_config.image_size,
                                           batch_size=model_config.predict_batch_size)

    start = time.time()
    y_pred = model.predict_generator(pre_flow, steps=len(pre_files) / model_config.predict_batch_size, verbose=1)
    print("####### predict %d images spend %d seconds ######"
          % (len(pre_files), time.time() - start))
    start = time.time()
    greedy_score, greedy_threshold = metrics.greedy_f2_score(y, y_pred, label_num=len(model_config.label_position))
    print("####### search greedy threshold spend %d seconds ######"
          % (time.time() - start))
    print("####### Smooth F2-Score is %f #######"
          % metrics.smooth_f2_score_np(y, y_pred))
    if len(model_config.label_position) > 1:
        print("####### F2-Score with threshold 0.2 is %f #######"
              % fbeta_score(y, (np.array(y_pred) > 0.2).astype(np.int8), beta=2, average='samples'))
        print("####### F2-Score with threshold 0.1 is %f #######"
              % fbeta_score(y, (np.array(y_pred) > 0.1).astype(np.int8), beta=2, average='samples'))
    else:
        print("####### F2-Score with threshold 0.2 is %f #######"
              % fbeta_score(y, (np.array(y_pred) > 0.2).astype(np.int8), beta=2))
        print("####### F2-Score with threshold 0.1 is %f #######"
              % fbeta_score(y, (np.array(y_pred) > 0.1).astype(np.int8), beta=2))

    print("####### Greedy F2-Score is %f #######" % greedy_score)

    with open(os.path.join(model_config.record_dir, "evaluate%s.txt" % str([str(i) for i in model_config.label_position])),
              "a") as f:
        f.write("\n\n")
        f.write("weight: %s\n" % weight_name)
        f.write("Smooth F2-Score: %f\n"
                % metrics.smooth_f2_score_np(y, y_pred))

        if len(model_config.label_position) > 1:
            f.write("F2-Score with threshold 0.2: %f\n"
                    % fbeta_score(y, (np.array(y_pred) > 0.2).astype(np.int8), beta=2, average='samples'))
            print("F2-Score with threshold 0.1: %f\n"
                  % fbeta_score(y, (np.array(y_pred) > 0.1).astype(np.int8), beta=2, average='samples'))
        else:
            f.write("F2-Score with threshold 0.2: %f\n"
                    % fbeta_score(y, (np.array(y_pred) > 0.2).astype(np.int8), beta=2))
            print("F2-Score with threshold 0.1: %f\n"
                  % fbeta_score(y, (np.array(y_pred) > 0.1).astype(np.int8), beta=2))

        f.write("Greedy F2-Score is: %f\n" % greedy_score)
        f.write("Greedy threshold: ")
        f.write(",".join([str(j) for j in greedy_threshold]))
        f.write("\n")

        greedy_threshold_all = []
        for i in range(y_pred.shape[-1]):
            smooth_f2 = metrics.smooth_f2_score_np(y[:, i], y_pred[:, i])
            greedy_f2, greedy_threshold = metrics.greedy_f2_score(y[:, i], y_pred[:, i], 1)
            bason_f2, bason_threshold = metrics.best_f2_score(y[:, i], y_pred[:, i], 1)
            print("[label %d]\tsmooth-f2=%4f   BFGS-f2=%4f[%4f]   greedy-f2=%4f[%4f]" % (
                model_config.label_position[i], smooth_f2, bason_f2, bason_threshold[0], greedy_f2,
                greedy_threshold[0]))
            f.write("[label %d]\tsmooth-f2=%4f   BFGS-f2=%4f[%4f]   greedy-f2=%4f[%4f]\n" % (
                model_config.label_position[i], smooth_f2, bason_f2, bason_threshold[0], greedy_f2,
                greedy_threshold[0]))
            greedy_threshold_all.append(greedy_threshold)
