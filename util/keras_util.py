import math
import os
import pathlib
import time

import keras
import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.callbacks import Callback
from sklearn.metrics import fbeta_score

from util import data_loader
from util import metrics
from util import path


class KerasModelConfig(object):
    def __init__(self,
                 k_fold_file,
                 model_path: str,
                 image_resolution,
                 data_type,
                 val_index=None,
                 input_norm=True,
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
        self.val_index = int("".join(filter(str.isdigit, file_name.split("_")[1]))) if val_index is None else val_index
        self.image_resolution = image_resolution
        self.image_size = (image_resolution, image_resolution)
        self.image_shape = (image_resolution, image_resolution, 3)
        self.input_norm = input_norm
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

        print("##########load model config")
        print("##########file name is: %s" % file_name)
        print("##########val index is: %d" % self.val_index)
        print("##########model dir is: %s" % model_dir)
        print("##########record dir is: %s" % self.record_dir)

    def get_steps_per_epoch(self):
        return math.ceil(len(self.train_files) / self.train_batch_size)

    def get_weights_path(self, epoch):
        if epoch < 10:
            return os.path.join(self.record_dir,
                                "%sweights.00%d.hdf5" % (str([str(j) for j in self.label_position]), epoch))
        else:
            return os.path.join(self.record_dir,
                                "%sweights.0%d.hdf5" % (str([str(j) for j in self.label_position]), epoch))


def predict(model: keras.Model, pre_files, model_config: KerasModelConfig, verbose=1):
    print("start predict")
    if model_config.input_norm:
        pre_datagen = data_loader.KerasGenerator(featurewise_center=True,
                                                 featurewise_std_normalization=True,
                                                 rescale=1. / 256)
    else:
        pre_datagen = data_loader.KerasGenerator()

    data_loader.check_mean_std_file(model_config, pre_datagen)
    pre_datagen.load_image_global_mean_std(model_config.image_mean_file, model_config.image_std_file)

    pre_flow = pre_datagen.flow_from_files(pre_files, mode="predict",
                                           target_size=model_config.image_size,
                                           batch_size=model_config.predict_batch_size)

    return model.predict_generator(pre_flow, steps=len(pre_files) / model_config.predict_batch_size, verbose=verbose)


def evaluate(y, y_pred, weight_name, model_config):
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

    with open(os.path.join(model_config.record_dir,
                           "evaluate%s.txt" % str([str(i) for i in model_config.label_position])),
              "a") as f:
        f.write("\n\n")
        f.write("weight: %s\n" % weight_name)
        f.write("Smooth F2-Score: %f\n"
                % metrics.smooth_f2_score_np(y, y_pred))

        if len(model_config.label_position) > 1:
            f.write("F2-Score with threshold 0.2: %f\n"
                    % fbeta_score(y, (np.array(y_pred) > 0.2).astype(np.int8), beta=2, average='samples'))
            f.write("F2-Score with threshold 0.1: %f\n"
                    % fbeta_score(y, (np.array(y_pred) > 0.1).astype(np.int8), beta=2, average='samples'))
        else:
            f.write("F2-Score with threshold 0.2: %f\n"
                    % fbeta_score(y, (np.array(y_pred) > 0.2).astype(np.int8), beta=2))
            f.write("F2-Score with threshold 0.1: %f\n"
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


def evaluate_model(model: keras.Model, pre_files, y, weight_name, model_config: KerasModelConfig):
    if y is None:
        y = np.array(data_loader.get_labels(model_config.val_files), np.bool)[:, model_config.label_position]

    if model_config.input_norm:
        pre_datagen = data_loader.KerasGenerator(featurewise_center=True,
                                                 featurewise_std_normalization=True,
                                                 rescale=1. / 256)
    else:
        pre_datagen = data_loader.KerasGenerator()

    data_loader.check_mean_std_file(model_config, pre_datagen)
    pre_datagen.load_image_global_mean_std(model_config.image_mean_file, model_config.image_std_file)

    pre_flow = pre_datagen.flow_from_files(pre_files, mode="predict",
                                           target_size=model_config.image_size,
                                           batch_size=model_config.predict_batch_size)

    start = time.time()
    y_pred = model.predict_generator(pre_flow, steps=len(pre_files) / model_config.predict_batch_size, verbose=1)
    print("####### predict %d images spend %d seconds ######"
          % (len(pre_files), time.time() - start))

    evaluate(y, y_pred, weight_name, model_config)


class CyclicLrCallback(Callback):
    """
    代码来源于：https://github.com/bckenstler/CLR.git
    This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each
        cycle iteration.
    For more detail, please see paper.

    # Example
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```

    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    """

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLrCallback, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1 / (2. ** (x - 1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma ** (x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.

    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(
                self.clr_iterations)

    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())

    def on_batch_end(self, epoch, logs=None):

        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        K.set_value(self.model.optimizer.lr, self.clr())


class TensorBoardCallback(keras.callbacks.TensorBoard):
    def __init__(self, log_every=1, model_config=None, **kwargs):
        super().__init__(**kwargs)
        self.log_every = log_every
        self.counter = 0
        self.model_config = model_config

    def on_epoch_begin(self, epoch, logs=None):
        self.counter = epoch * self.model_config.get_steps_per_epoch()
        print("on epoch begin, set counter %f" % self.counter)

    def on_batch_end(self, batch, logs=None):
        self.counter += 1
        if self.counter % self.log_every == 0:
            for name, value in logs.items():
                if name in ['batch', 'size']:
                    continue
                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = value.item()
                summary_value.tag = "batch/" + name
                self.writer.add_summary(summary, self.counter)
            self.writer.flush()

        super().on_batch_end(batch, logs)
