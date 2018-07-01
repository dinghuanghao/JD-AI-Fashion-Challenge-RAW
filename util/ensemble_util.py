import json
import os
import pathlib
import time

from statistics import model_statistics as statis


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
                 all_label=True,
                 debug=False
                 ):
        file_name = os.path.basename(model_path)
        model_dir = os.path.dirname(model_path)
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
