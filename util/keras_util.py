import math
import os
import pathlib
import random
import time
from queue import Queue
from threading import Thread

import keras
import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.callbacks import Callback
from sklearn.metrics import fbeta_score

from util import data_loader
from util import data_visualization as dv
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
                 downsampling=None,
                 data_visualization=False,
                 label_position=(1,),
                 label_color_augment=None,
                 label_up_sampling=None,
                 train_batch_size=(32,),
                 val_batch_size=32,
                 predict_batch_size=32,
                 epoch=(1,),
                 initial_epoch=0,
                 lr=(0.01,),
                 freeze_layers=(0,),
                 tta=False,
                 debug=False):

        file_name = os.path.basename(model_path)
        model_dir = os.path.dirname(model_path)

        self.k_fold_file = k_fold_file
        self.model_path = model_path
        self.val_index = int("".join(filter(str.isdigit, file_name.split("_")[1]))) if val_index is None else val_index
        self.image_resolution = image_resolution
        self.image_size = (image_resolution, image_resolution)
        self.image_shape = (image_resolution, image_resolution, 3)
        self.input_norm = input_norm
        self.data_type = data_type
        self.record_dir = os.path.join(os.path.join(model_dir, "record"), file_name.split("_")[0])
        self.record_dir = os.path.join(self.record_dir, "val%d" % self.val_index)
        self.fit_img_record_dir = os.path.join(os.path.join(self.record_dir, "image"), "fit")
        self.predict_img_record_dir = os.path.join(os.path.join(self.record_dir, "image"), "predict")
        self.log_file = os.path.join(self.record_dir, "log.txt")
        self.label_position = label_position
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.predict_batch_size = predict_batch_size
        self.epoch = epoch
        self.initial_epoch = initial_epoch
        self.lr = lr
        self.freeze_layers = freeze_layers
        self.writer = tf.summary.FileWriter(self.record_dir)
        self.current_epoch = initial_epoch
        self.tta = tta
        self.debug = debug
        self.data_visualization = data_visualization

        self.val_files = []
        self.train_files = []

        for i in self.data_type:
            train_files, val_files = data_loader.get_k_fold_files(self.k_fold_file, self.val_index, [i])
            self.val_files.append(val_files)
            self.train_files += train_files

        if label_color_augment is not None:

            # 将当前用于train的所有图片名称构成一个dict
            train_file_dict = {}
            for train_file in self.train_files:
                train_file_dict.setdefault(os.path.split(train_file)[-1], None)

            from preprocess.augment import color
            augment_image_dirs = color.get_augment_image_dirs()
            labels = data_loader.get_labels(augment_image_dirs)

            augment_files = []
            for i in range(len(augment_image_dirs)):
                # 如果augment的数据名称不在当前train集中，说明是val数据，跳过
                if os.path.split(augment_image_dirs[i])[-1] not in train_file_dict:
                    continue
                label = labels[i]
                for j in label_color_augment:
                    if label[j] == 1:
                        augment_files.append(augment_image_dirs[i])
                        break
            self.train_files += augment_files
            self.save_log("add %d color augmentation file" % len(augment_files))

        if label_up_sampling is not None:
            self.save_log("train files is %d before up sampling" % len(self.train_files))
            sampling_files = []
            sampling_times = [0 for i in range(13)]
            labels = data_loader.get_labels(self.train_files)
            for i in range(len(labels)):
                for j in range(len(label_up_sampling)):
                    label = labels[i]
                    if label[j] > 0 and label_up_sampling[j] > 0:
                        sampling_files += [self.train_files[i]] * label_up_sampling[j]
                        sampling_times[j] += label_up_sampling[j]

            self.train_files += sampling_files
            self.save_log(
                "up sampling times: %s, totaol: %d" % (str([str(i) for i in sampling_times]), sum(sampling_times)))
            self.save_log("train files is %d after up sampling" % len(self.train_files))

        self.val_y = np.array(data_loader.get_labels(self.val_files[0]), np.bool)[:, self.label_position]
        if downsampling is not None:
            new_train_files = []
            for _ in self.train_files:
                _label = data_loader.get_label(_.split(os.sep)[-1])
                if _label == ['0', '0', '1', '0', '0', '0', '0', '0', '1', '0', '0', '0',
                              '0'] and random.random() > downsampling:
                    _label = data_loader.get_label(_.split(os.sep)[-1])
                _labels = [
                    ['0', '0', '1', '0', '0', '0', '0', '0', '1', '0', '0', '0', '0'],
                    ['0', '0', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0'],
                    ['0', '0', '1', '0', '1', '0', '0', '0', '1', '0', '0', '0', '0'],
                    ['0', '0', '0', '0', '0', '0', '0', '0', '1', '0', '0', '0', '0']
                ]
                if _label in _labels and random.random() > downsampling:
                    continue
                else:
                    new_train_files.append(_)
            self.train_files = new_train_files

        random.shuffle(self.train_files)
        self.train_files = np.array(self.train_files)

        if self.data_visualization:
            dv.show_label_calss_bar_per_epoch(self.train_files, self.record_dir)

        if debug:
            self.train_files = self.train_files[:64]
            for i in range(len(self.val_files)):
                self.val_files[i] = self.val_files[i][:64]
            self.val_y = self.val_y[:64]

        self.image_mean_file = path.get_image_mean_file(self.k_fold_file, self.val_index,
                                                        data_type=self.data_type)
        self.image_std_file = path.get_image_std_file(self.k_fold_file, self.val_index,
                                                      data_type=self.data_type)

        self.save_model_format = os.path.join(self.record_dir,
                                              "%sweights.{epoch:03d}.hdf5" % str([str(i) for i in self.label_position]))
        self.tem_model_file = os.path.join(self.record_dir, 'weights.hdf5')
        pathlib.Path(self.record_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.fit_img_record_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.predict_img_record_dir).mkdir(parents=True, exist_ok=True)

        print("##########load model config")
        print("##########file name is: %s" % file_name)
        print("##########val index is: %d" % self.val_index)
        print("##########model dir is: %s" % model_dir)
        print("##########record dir is: %s" % self.record_dir)
        self.save_log("train file: %d, val file: %d" % (len(self.train_files), len(self.val_y)))

    def save_log(self, log):
        log = time.strftime("%Y-%m-%d:%H:%M:%S") + ": " + log
        print(log)
        with open(self.log_file, "a") as f:
            f.write(log)
            f.write("\n")

    def decrease_train_files(self, num):
        self.train_files = self.train_files[:num]

    def decrease_val_files(self, num):
        for i in range(len(self.val_files)):
            self.val_files[i] = self.val_files[i][:num]

        self.val_y = self.val_y[:num]

    def get_init_stage(self):
        stage = 0
        for i in range(len(self.epoch)):
            stage = i
            if self.initial_epoch + 1 <= self.epoch[i]:
                break
        return stage

    def get_stage(self, epoch):
        stage = 0
        for i in range(len(self.epoch)):
            stage = i
            if epoch + 1 <= self.epoch[i]:
                break
        return stage

    def get_steps_per_epoch(self, stage):
        return math.ceil(len(self.train_files) / self.train_batch_size[stage])

    def get_weights_path(self, epoch):
        return os.path.join(self.record_dir,
                            "%sweights.%03d.hdf5" % (str([str(j) for j in self.label_position]), epoch))


def predict_tta(model: keras.Model, model_config: KerasModelConfig, verbose=1):
    if model_config.input_norm:
        pre_datagen = data_loader.KerasGenerator(featurewise_center=True,
                                                 featurewise_std_normalization=True,
                                                 rescale=1. / 256,
                                                 model_config=model_config,
                                                 real_transform=True)
    else:
        pre_datagen = data_loader.KerasGenerator(model_config=model_config, real_transform=True)

    pre_datagen.check_mean_std_file(model_config)
    pre_datagen.load_image_global_mean_std(model_config.image_mean_file, model_config.image_std_file)

    y_pred = None
    start = time.time()

    tta = data_loader.TestTimeAugmentation()
    pre_datagen.tta = tta
    predict_times = 0

    for files in model_config.val_files:
        for i in range(tta.tta_times):
            predict_times += 1
            model_config.save_log("start predict with tta index is %d" % i)
            pre_flow = pre_datagen.flow_from_files(files, mode="predict",
                                                   target_size=model_config.image_size,
                                                   batch_size=model_config.predict_batch_size,
                                                   tta_index=i)

            if y_pred is None:
                y_pred = np.array(model.predict_generator(pre_flow, steps=len(files) / model_config.predict_batch_size,
                                                          verbose=verbose, workers=16))
            else:
                y_pred += np.array(model.predict_generator(pre_flow, steps=len(files) / model_config.predict_batch_size,
                                                           verbose=verbose, workers=16))

    assert y_pred.shape[0] == model_config.val_y.shape[0]

    y_pred = y_pred / (len(model_config.data_type) * tta.tta_times)
    print("####### predict %d times, spend %d seconds total ######" % (predict_times, time.time() - start))
    return y_pred


def predict(model: keras.Model, model_config: KerasModelConfig, verbose=1):
    if model_config.input_norm:
        pre_datagen = data_loader.KerasGenerator(featurewise_center=True,
                                                 featurewise_std_normalization=True,
                                                 rescale=1. / 256)
    else:
        pre_datagen = data_loader.KerasGenerator()

    pre_datagen.check_mean_std_file(model_config)
    pre_datagen.load_image_global_mean_std(model_config.image_mean_file, model_config.image_std_file)

    y_pred = None
    start = time.time()
    for files in model_config.val_files:
        pre_flow = pre_datagen.flow_from_files(files, mode="predict",
                                               target_size=model_config.image_size,
                                               batch_size=model_config.predict_batch_size)

        if y_pred is None:
            y_pred = np.array(model.predict_generator(pre_flow, steps=len(files) / model_config.predict_batch_size,
                                                      verbose=verbose, workers=16))
        else:
            y_pred += np.array(model.predict_generator(pre_flow, steps=len(files) / model_config.predict_batch_size,
                                                       verbose=verbose, workers=16))

    assert y_pred.shape[0] == model_config.val_y.shape[0]

    y_pred = y_pred / len(model_config.data_type)
    print("####### predict spend %d seconds ######" % (time.time() - start))
    return y_pred


def summary_val_value(name, value, model_config):
    summary = tf.Summary()
    summary_value = summary.value.add()
    summary_value.simple_value = value
    summary_value.tag = name
    model_config.writer.add_summary(summary, model_config.current_epoch)
    model_config.writer.flush()


def evaluate(y, y_pred, weight_name, model_config: KerasModelConfig):
    if len(model_config.label_position) > 1:
        thread_f2_01 = fbeta_score(y, (np.array(y_pred) > 0.1).astype(np.int8), beta=2, average='macro')
        thread_f2_02 = fbeta_score(y, (np.array(y_pred) > 0.2).astype(np.int8), beta=2, average='macro')
    else:
        thread_f2_01 = fbeta_score(y, (np.array(y_pred) > 0.1).astype(np.int8), beta=2)
        thread_f2_02 = fbeta_score(y, (np.array(y_pred) > 0.2).astype(np.int8), beta=2)

    one_label_greedy_f2_all = []
    one_label_greedy_threshold_all = []
    one_label_smooth_f2_all = []
    for i in range(y.shape[-1]):
        one_label_smooth_f2 = metrics.smooth_f2_score_np(y[:, i], y_pred[:, i])
        one_label_greedy_f2, greedy_threshold = metrics.greedy_f2_score(y[:, i], y_pred[:, i], 1)
        one_label_smooth_f2_all.append(one_label_smooth_f2)
        one_label_greedy_f2_all.append(one_label_greedy_f2)
        one_label_greedy_threshold_all.append(greedy_threshold[0])

    print("####### Smooth F2-Score is %6f #######" % np.mean(one_label_smooth_f2_all))
    print("####### F2-Score with threshold 0.1 is %6f #######" % thread_f2_01)
    print("####### F2-Score with threshold 0.2 is %6f #######" % thread_f2_02)
    print("####### Greedy F2-Score is %6f #######" % np.mean(one_label_greedy_f2_all))

    summary_val_value("val-label-all/smooth-f2", np.mean(one_label_smooth_f2_all), model_config)
    summary_val_value("val-label-all/thread-f2-01", thread_f2_01, model_config)
    summary_val_value("val-label-all/thread-f2-02", thread_f2_02, model_config)
    summary_val_value("val-label-all/greedy-f2", np.mean(one_label_greedy_f2_all), model_config)

    for i in range(len(one_label_greedy_f2_all)):
        print("[label %d]\tsmooth-f2=%4f greedy-f2=%4f[%4f]" % (
            model_config.label_position[i], one_label_smooth_f2_all[i], one_label_greedy_f2_all[i],
            one_label_greedy_threshold_all[i]))

        summary_val_value("val-label-%d/smooth-f2" % model_config.label_position[i], one_label_smooth_f2_all[i],
                          model_config)
        summary_val_value("val-label-%d/greedy-f2" % model_config.label_position[i], one_label_greedy_f2_all[i],
                          model_config)

    with open(os.path.join(model_config.record_dir,
                           "evaluate%s.txt" % str([str(i) for i in model_config.label_position])), "a") as f:
        f.write("\n\n")
        f.write("Weight: %s\n" % weight_name)
        f.write("Smooth F2-Score: %f\n" % np.mean(one_label_smooth_f2_all))
        f.write("F2-Score with threshold 0.1: %f\n" % thread_f2_01)
        f.write("F2-Score with threshold 0.2: %f\n" % thread_f2_02)
        f.write("Greedy F2-Score is: %f\n" % np.mean(one_label_greedy_f2_all))

        for i in range(len(one_label_greedy_f2_all)):
            f.write("[label %d]\tsmooth-f2=%4f   greedy-f2=%4f[%4f]\n" % (
                model_config.label_position[i], one_label_smooth_f2_all[i], one_label_greedy_f2_all[i],
                one_label_greedy_threshold_all[i]))


def get_prediction_path(weight_path):
    return weight_path + ".predict.npy"


def save_prediction_file(prediction, weight_path, overwrite=False):
    prediction_path = get_prediction_path(weight_path)
    if os.path.exists(prediction_path):
        if overwrite:
            print("overwrite prediction: %s" % prediction_path)
            np.save(prediction_path, prediction)
    else:
        print("save prediction: %s" % prediction_path)
        np.save(prediction_path, prediction)


class EvaluateTask(Thread):
    def __init__(self, q: Queue):
        Thread.__init__(self)
        self.q = q

    def run(self):
        while True:
            y, y_pred, weight_path, model_config = self.q.get()
            print("start evaluate task for %s" % weight_path)
            save_prediction_file(y_pred, weight_path)
            evaluate(y, y_pred, weight_path, model_config)


class EvaluateCallback(Callback):
    def __init__(self, model_config: KerasModelConfig, evaluate_queue: Queue):
        super(EvaluateCallback, self).__init__()
        self.model_config = model_config
        self.evaluate_queue = evaluate_queue

    def on_epoch_end(self, epoch, logs=None):
        real_epoch = epoch + 1
        self.model_config.current_epoch = real_epoch
        self.model_config.save_log(
            "on epoch %d end, save weight:%s" % (real_epoch, self.model_config.get_weights_path(real_epoch)))
        self.model.save_weights(self.model_config.get_weights_path(real_epoch), overwrite=True)
        if self.model_config.tta:
            self.model_config.save_log("start predict with tta")
            y_pred = predict_tta(self.model, self.model_config, verbose=1)
        else:
            self.model_config.save_log("start predict")
            y_pred = predict(self.model, self.model_config, verbose=1)
        self.evaluate_queue.put(
            (self.model_config.val_y, y_pred, self.model_config.get_weights_path(real_epoch), self.model_config))


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
        stage = self.model_config.get_stage(epoch)
        self.counter = epoch * self.model_config.get_steps_per_epoch(stage)
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
