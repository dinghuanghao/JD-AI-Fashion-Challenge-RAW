import os

from statistics import model_statistics as statis
from util import path


class EnsembleConfig(object):
    """
    1. 将选择的模型保存到本地，且如果本地有模型选择记录，则直接使用本地记录
    2. 只选择与输出相同的标签（N * 1），还是所有标签（N * 13）
    """

    def __init__(self,
                 model_path: str,
                 corr_threshold=0.9,
                 search=20,
                 top_n=5,
                 all_label=False,
                 debug=False
                 ):
        file_name = os.path.basename(model_path)
        model_dir = os.path.dirname(model_path)
        self.record_dir = os.path.join(os.path.join(model_dir, "record"), file_name)
        self.meta_model_file = os.path.join(self.record_dir, "meta_model.txt")

        self.meta_model_all = []

        for val_index in range(1, 6):
            all_label, one_label, thresholds = statis.model_f2_statistics(path.MODEL_PATH, val_index)
            all_label, one_label, thresholds = statis.model_f2_statistics_no_repeat(all_label, one_label, thresholds)
            meta_model_val = [[] for i in range(13)]
            for label in range(13):
                corr = statis.model_coor(one_label[label][:search], label, thresholds, val_index).values
                ignore = []
                for i in range(search):
                    for j in range(search):
                        if corr[i, j] > corr_threshold and i not in ignore and i != j:
                            ignore.append(j)

                for i in range(len(one_label[label])):
                    if i in ignore:
                        continue
                    meta_model_val[label].append(one_label[label][i])
                    if len(meta_model_val) >= top_n:
                        break

            self.meta_model_all.append(meta_model_val)

    def save_meta_model(self):
        with open(self.meta_model_file, "w+") as f:
            for val in range(5):
                for label in range(13):
                    meta_model = self.meta_model_all[val][label]
                    f.write("[val %d][label %d][f2 %4f]: %s" % (val + 1, label, meta_model[1], meta_model[0]))

    def get_meta_model(self):
        if not os.path.exists(self.meta_model_file):
            return None

        with open(self.meta_model_file, "as ") as f:


class ModelEnsemble(object):
    """
    1. 自动做K-FOLD训练，将所有的模型都进行保存， 并将预测结果和评估结果也进行保存
    """

    def __init__(self, config: EnsembleConfig):
        pass

    def train_all_label(self):
        pass

    def train_single_label(self, label):
        pass
