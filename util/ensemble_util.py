import copy
import json
import os
import pathlib
import time

import numpy as np
import xgboost as xgb
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
        self.evaluate_json = os.path.join(self.record_dir, "evaluate.json")

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

        with open(self.meta_mode_json, "r") as f:
            return json.load(f)

    def save_evaluate_json(self, evaluate):
        with open(self.evaluate_json, "w+") as f:
            json.dump(evaluate, f)

    def get_evaluate_json(self):
        if not os.path.exists(self.evaluate_json):
            return {}

        with open(self.evaluate_json, "r") as f:
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

    def evaluate(self, y, y_pred, weight_name, xgb_param=None, save_evaluate=False):
        if y.shape[1] > 1:
            thread_f2_01 = fbeta_score(y, (np.array(y_pred) > 0.1).astype(np.int8), beta=2, average='macro')
            thread_f2_02 = fbeta_score(y, (np.array(y_pred) > 0.2).astype(np.int8), beta=2, average='macro')
        else:
            thread_f2_01 = fbeta_score(y, (np.array(y_pred) > 0.1).astype(np.int8), beta=2)
            thread_f2_02 = fbeta_score(y, (np.array(y_pred) > 0.2).astype(np.int8), beta=2)

        one_label_greedy_f2_all = []
        one_label_greedy_threshold_all = []
        one_label_smooth_f2_all = []
        assert y.shape[-1] == 1
        for i in range(y.shape[-1]):
            one_label_smooth_f2 = metrics.smooth_f2_score_np(y[:, i], y_pred[:, i])
            one_label_greedy_f2, greedy_threshold = metrics.greedy_f2_score(y[:, i], y_pred[:, i], 1)
            one_label_smooth_f2_all.append(one_label_smooth_f2)
            one_label_greedy_f2_all.append(one_label_greedy_f2)
            one_label_greedy_threshold_all.append(greedy_threshold[0])

        greedy_f2 = np.mean(one_label_greedy_f2_all)

        print("####### Smooth F2-Score is %6f #######" % np.mean(one_label_smooth_f2_all))
        print("####### F2-Score with threshold 0.1 is %6f #######" % thread_f2_01)
        print("####### F2-Score with threshold 0.2 is %6f #######" % thread_f2_02)
        print("####### Greedy F2-Score is %6f #######" % greedy_f2)

        if save_evaluate:
            evaluate = self.get_evaluate_json()
            evaluate[weight_name] = {}
            evaluate[weight_name]['eta'] = xgb_param['eta']
            evaluate[weight_name]['max_depth'] = xgb_param['max_depth']
            evaluate[weight_name]['min_child_weight'] = xgb_param['min_child_weight']
            evaluate[weight_name]['best_iteration'] = xgb_param['best_iteration']
            evaluate[weight_name]['best_ntree_limit'] = xgb_param['best_ntree_limit']
            evaluate[weight_name]['smooth_f2'] = np.mean(one_label_smooth_f2_all)
            evaluate[weight_name]['f2_0.1'] = thread_f2_01
            evaluate[weight_name]['f2_0.2'] = thread_f2_02
            evaluate[weight_name]['greedy_threshold'] = one_label_greedy_threshold_all[0]
            evaluate[weight_name]['greedy_f2'] = greedy_f2
            self.save_evaluate_json(evaluate)

        return greedy_f2


def xgb_f2_metric(preds, dtrain):  # preds是结果（概率值），dtrain是个带label的DMatrix
    labels = dtrain.get_label()  # 提取label
    thread_f2_02 = fbeta_score(labels, (np.array(preds) > 0.2).astype(np.int8), beta=2)
    return 'F2-0.2', 1 - thread_f2_02


def xgb_greedy_f2_metric(preds, dtrain, step=100):
    labels = dtrain.get_label()  # 提取label
    thread_f2_02, _ = metrics.greedy_f2_score(labels, preds, 1, step)
    return 'Greedy-F2', 1 - thread_f2_02


class XGBoostModel(EnsembleModel):
    def __init__(self, xgb_param:dict=None, number_round=None, eval_func=None,
                 *args, **kwargs):
        super(XGBoostModel, self).__init__(*args, **kwargs)
        self.xgb_param = xgb_param
        self.number_round = number_round
        self.best_ntree_json = os.path.join(self.record_dir, "best_ntree.json")
        self.model_dir = os.path.join(self.record_dir, "booster")
        if eval_func is None:
            self.eval_func = xgb_f2_metric
        else:
            self.eval_func = eval_func

        pathlib.Path(self.model_dir).mkdir(parents=True, exist_ok=True)

        if self.xgb_param.get('min_child_weight', None) is None:
            self.xgb_param['min_child_weight'] = [1]
        if self.xgb_param.get('eta', None) is None:
            self.xgb_param['eta'] = [0.3]
        if self.xgb_param.get('max_depth', None) is None:
            self.xgb_param['max_depth'] = [6]

    def load_model(self, val_index, label):
        booster = xgb.Booster()
        booster.load_model(os.path.join(self.model_dir, self.get_model_name(val_index, label)))
        return booster

    def save_model(self, model, val_index, label):
        model.save_model(os.path.join(self.model_dir, self.get_model_name(val_index, label)))

    def get_model_name(self, val_index, label):
        return "ensemble_val%d_label%d.xgb" % (val_index, label)

    def train_all_label(self):
        for val_index in range(1, 6):
            for label in range(13):
                self.train_single_label(val_index=val_index, label=label)

    def train_single_label(self, val_index, label):
        train_x, train_y, val_x, val_y = self.build_datasets(val_index=val_index, target_label=label)
        data_train = xgb.DMatrix(data=train_x, label=train_y)
        data_val = xgb.DMatrix(data=val_x, label=val_y)

        evals = [(data_train, 'train'), (data_val, 'eval')]
        best_f2 = 0
        best_eta = 0
        best_max_depth = 0
        best_model = None
        best_pred = None
        best_xgb_param = None
        best_min_child_weight = None

        for eta in self.xgb_param["eta"]:
            for max_depth in self.xgb_param['max_depth']:
                for min_child_weight in self.xgb_param['min_child_weight']:
                    xgb_param = {
                        'eta': eta,
                        'silent': self.xgb_param['silent'],  # option for logging
                        'objective': self.xgb_param['objective'],  # error evaluation for multiclass tasks
                        'max_depth': max_depth,  # depth of the trees in the boosting process
                        'min_child_weight': min_child_weight
                    }

                    bst = xgb.train(xgb_param, data_train, self.number_round, evals=evals,
                                    feval=self.eval_func,
                                    early_stopping_rounds=10)

                    data_eva = xgb.DMatrix(val_x)
                    ypred = bst.predict(data_eva, ntree_limit=bst.best_ntree_limit)
                    ypred = ypred.reshape((-1, 1))
                    f2 = self.evaluate(y_pred=ypred, y=val_y, weight_name=self.get_model_name(val_index, label))
                    self.save_log("eta:%f, max_depth:%d, f2:%f" % (eta, max_depth, f2))
                    self.save_log("best_iteration:%4f,  best_score:%4f, best_ntree_limit=%4f" % (bst.best_iteration,
                                                                                                 bst.best_score,
                                                                                                 bst.best_ntree_limit))
                    self.save_log("\n")

                    if f2 > best_f2:
                        best_f2 = f2
                        best_model = bst
                        best_eta = eta
                        best_max_depth = max_depth
                        best_min_child_weight = min_child_weight
                        best_pred = ypred
                        best_xgb_param = copy.deepcopy(xgb_param)
                        best_xgb_param['best_ntree_limit'] = bst.best_ntree_limit
                        best_xgb_param['best_iteration'] = bst.best_iteration

        self.evaluate(y_pred=best_pred, y=val_y, weight_name=self.get_model_name(val_index, label),
                      xgb_param=best_xgb_param, save_evaluate=True)

        self.save_log(
            "save best model for val[%d] label[%d], f2[%f] eta[%f] max_depth[%d]  best_min_child_weight[%f] best_ntree[%d] best_iter[%d]" %
            (val_index, label, best_f2, best_eta, best_max_depth, best_min_child_weight, best_xgb_param['best_ntree_limit'],
             best_xgb_param['best_iteration']))

        self.save_model(best_model, val_index, label)

        # 测试load_model是否正确
        model = self.load_model(val_index, label)
        data_eva = xgb.DMatrix(val_x)
        ypred = model.predict(data_eva, ntree_limit=best_xgb_param['best_ntree_limit'])
        ypred = ypred.reshape((-1, 1))
        f2 = self.evaluate(y_pred=ypred, y=val_y, weight_name=self.get_model_name(val_index, label))
        assert abs((f2 - best_f2) / f2) < 0.001

    def predict_all_label(self):
        for val_index in range(1, 6):
            for label in range(13):
                self.predict_one_label(val_index, label)

    def predict_one_label(self, val_index, label, ntree_limit):
        bst = self.load_model(val_index, label)
