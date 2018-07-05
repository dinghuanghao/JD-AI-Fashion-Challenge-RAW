import json
import os
import pathlib
import time

import numpy as np
from sklearn.metrics import fbeta_score

from statistics import model_statistics as statis
from util import data_loader
from util import keras_util
from util import metrics


class EnsembleModel(object):
    """
    1. 自动做K-FOLD训练，将所有的模型都进行保存， 并将预测结果和评估结果也进行保存
    """

    def __init__(self,
                 model_path: str,
                 corr_threshold=0.9,
                 search=20,
                 top_n=5,
                 debug=False
                 ):
        file_name = os.path.basename(model_path)
        model_dir = os.path.dirname(model_path)
        self.dataset = []
        self.corr_threshold = corr_threshold
        self.search = search
        self.top_n = top_n
        self.record_dir = os.path.join(os.path.join(model_dir, "record"), file_name.split(".")[0])
        self.statistics_dir = os.path.join(self.record_dir, "statistics")
        self.log_file = os.path.join(self.record_dir, "log.txt")
        self.meta_model_txt = os.path.join(self.record_dir, "meta_model.txt")
        self.meta_mode_json = os.path.join(self.record_dir, "meta_model.json")

        pathlib.Path(self.record_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.statistics_dir).mkdir(parents=True, exist_ok=True)

        self.meta_model_all = self.get_meta_model()
        if self.meta_model_all is None:
            self.meta_model_all = []
            one_label_all, corr_all = statis.do_statistics(self.statistics_dir, search)

            for val_index in range(5):
                self.save_log("start search val %d" % (val_index + 1))
                one_label = one_label_all[val_index]
                meta_model_val = [[] for i in range(13)]
                for label in range(13):
                    self.save_log("start search label %d" % label)
                    _search = min(search, len(one_label[label]))
                    ignore = []
                    corr = corr_all[val_index][label].values
                    for i in range(_search):
                        for j in range(_search):
                            if corr[i, j] > corr_threshold and i not in ignore and i != j:
                                self.save_log("ignore[corr %3f] %s" % (corr[i, j], one_label[label][j][0]))
                                ignore.append(j)

                    for i in range(len(one_label[label])):
                        if i in ignore:
                            continue
                        meta_model_val[label].append(one_label[label][i])
                        if len(meta_model_val[label]) >= top_n:
                            break

                    assert len(meta_model_val[label]) == top_n

                self.meta_model_all.append(meta_model_val)

            self.save_meta_model()

            if debug:
                saved_meta_model = self.get_meta_model()
                for val in range(5):
                    for label in range(13):
                        for i in range(top_n):
                            assert self.meta_model_all[val][label][i][0] == saved_meta_model[val][label][i][0]
                            assert self.meta_model_all[val][label][i][1] == saved_meta_model[val][label][i][1]
        else:
            self.save_log("load meta model info")

    def save_log(self, log):
        log = time.strftime("%Y-%m-%d:%H:%M:%S") + ": " + log
        print(log)
        with open(self.log_file, "a") as f:
            f.write(log)
            f.write("\n")

    def save_meta_model(self):
        with open(self.meta_model_txt, "w+") as f:
            for val in range(5):
                f.write("##############val %d###############\n" % val)
                for label in range(13):
                    f.write("--------------label %d--------------\n" % label)
                    for meta_model in self.meta_model_all[val][label]:
                        f.write("[f2 %4f]:%s\n" % (meta_model[1], meta_model[0]))
        with open(self.meta_mode_json, "w+") as f:
            json.dump(self.meta_model_all, f)

    def get_meta_model(self):
        if not os.path.exists(self.meta_model_txt):
            return None

        # TODO: 将文件路径进行转换
        with open(self.meta_mode_json, "r") as f:
            return json.load(f)

    def build_datasets(self, val_index, target_label, train_label=None):
        assert len(self.meta_model_all) == 5

        train_x, val_x = None, None

        if train_label is None:
            labels = [i for i in range(13)]
        else:
            labels = train_label

        samples_cnt = 0
        for val in range(1, 6):
            meta_model_val = self.meta_model_all[val - 1]
            predict_val = None
            assert len(meta_model_val) == 13
            for label in labels:
                meta_model_label = meta_model_val[label]
                for meta_model in meta_model_label:
                    predicts = np.load(keras_util.get_prediction_path(meta_model[0]))
                    predict_label = predicts[:, label].reshape((-1, 1))
                    samples_cnt += predict_label.shape[0]
                    if predict_val is None:
                        predict_val = predict_label
                    else:
                        predict_val = np.hstack((predict_val, predict_label))

            if val == val_index:
                val_x = np.copy(predict_val)
            else:
                if train_x is None:
                    train_x = np.copy(predict_val)
                else:
                    train_x = np.vstack((train_x, predict_val))

            assert predict_val.shape[1] == len(labels) * self.top_n

        train_y, val_y = data_loader.get_k_fold_labels(val_index, target_label)

        return train_x.astype(np.float32), train_y.astype(np.float32), val_x.astype(np.float32), val_y.astype(
            np.float32)

    def train_all_label(self):
        pass

    def train_single_label(self, val_index, label):
        pass

    def evaluate(self, y, y_pred, weight_name):
        if y.shape[1] > 1:
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

        # summary_val_value("val-label-all/smooth-f2", np.mean(one_label_smooth_f2_all), model_config)
        # summary_val_value("val-label-all/thread-f2-01", thread_f2_01, model_config)
        # summary_val_value("val-label-all/thread-f2-02", thread_f2_02, model_config)
        # summary_val_value("val-label-all/greedy-f2", np.mean(one_label_greedy_f2_all), model_config)
        #
        # for i in range(len(one_label_greedy_f2_all)):
        #     print("[label %d]\tsmooth-f2=%4f greedy-f2=%4f[%4f]" % (
        #         model_config.label_position[i], one_label_smooth_f2_all[i], one_label_greedy_f2_all[i],
        #         one_label_greedy_threshold_all[i]))

        # summary_val_value("val-label-%d/smooth-f2" % model_config.label_position[i], one_label_smooth_f2_all[i],
        #                   model_config)
        # summary_val_value("val-label-%d/greedy-f2" % model_config.label_position[i], one_label_greedy_f2_all[i],
        #                   model_config)

        with open(os.path.join(self.record_dir,
                               "evaluate.txt"), "a") as f:
            f.write("\n\n")
            f.write("Weight: %s\n" % weight_name)
            f.write("Smooth F2-Score: %f\n" % np.mean(one_label_smooth_f2_all))
            f.write("F2-Score with threshold 0.1: %f\n" % thread_f2_01)
            f.write("F2-Score with threshold 0.2: %f\n" % thread_f2_02)
            f.write("Greedy F2-Score is: %f\n" % np.mean(one_label_greedy_f2_all))

            # for i in range(len(one_label_greedy_f2_all)):
            #     f.write("[label %d]\tsmooth-f2=%4f   greedy-f2=%4f[%4f]\n" % (
            #         model_config.label_position[i], one_label_smooth_f2_all[i], one_label_greedy_f2_all[i],
            #         one_label_greedy_threshold_all[i]))


def xgb_f2_metric(preds, dtrain):  # preds是结果（概率值），dtrain是个带label的DMatrix
    labels = dtrain.get_label()  # 提取label
    thread_f2_02 = fbeta_score(labels, (np.array(preds) > 0.2).astype(np.int8), beta=2)
    return 'F2-0.2', 1 - thread_f2_02
