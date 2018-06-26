import os
import re

import numpy as np
import pandas as pd
import seaborn as sns

from util import path

RECORD_DIR = os.path.join(os.path.abspath("."), "record")

def model_f2_statistics(path, val_index=1, save_file=None):
    """
    对model目录下的所有包含"evaluate"字段的文件进行统计，分别得到all-label、one-label统计
    :param path: 需要统计的目录
    :param save_file: 输入文件
    :return:
    """
    evaluate_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if ("val%d" % val_index) not in root:
                continue
            if "evaluate" in file and "evaluate_revise" not in file:
                evaluate_files.append(os.path.join(root, file))

    all_label = {}
    one_label = {}
    for file in evaluate_files:
        with open(file, "r") as f:
            weight_file = ""
            for i in f.readlines():
                if "Weight" in i:
                    weight_file = re.match(r"Weight: *(.*)", i).group(1)

                if "Greedy F2-Score is:" in i:
                    if weight_file == "":
                        print("file %s is abnormal" % file)
                    greedy_f2 = i.split(":")[-1].strip()
                    all_label[weight_file] = float(greedy_f2)
                if "[label" in i:
                    if weight_file == "":
                        print("file %s is abnormal" % file)
                    if weight_file == "":
                        print("a")
                    label = re.match(r".*label *([0-9]*)", i).group(1)
                    greedy_f2 = re.match(r".*greedy-f2=(.*)\[", i).group(1)
                    if one_label.get(int(label), None) is None:
                        one_label[int(label)] = {}
                    one_label[int(label)][weight_file] = float(greedy_f2)

    all_label = sorted(all_label.items(), key=lambda x: x[1], reverse=True)
    for i in range(len(one_label)):
        one_label[i] = sorted(one_label[i].items(), key=lambda x: x[1], reverse=True)

    if save_file is not None:
        with open(save_file, "w") as f:
            f.write("==========================All label==========================\n")
            for i in all_label:
                f.write("%f: %s\n" % (i[1], i[0]))

            for i in range(len(one_label)):
                f.write("\n\n\n\n\n==========================One label: %d==========================\n" % i)
                for j in one_label[i]:
                    f.write("%f: %s\n" % (j[1], j[0]))

    return all_label, one_label


def path_2_model_name(weight_path):
    tmp = os.path.split(weight_path)
    weight_name = tmp[-1]
    epoch = weight_name.split('.')[1]
    tmp = os.path.split(tmp[0])
    val = tmp[-1]
    tmp = os.path.split(tmp[0])
    model_number = tmp[-1]
    tmp = os.path.split(tmp[0])
    tmp = os.path.split(tmp[0])
    model_type = tmp[-1]

    return "%s_%s_%s_%s" % (model_type, model_number, val, epoch), (model_type, model_number, val, epoch)


def model_corr_heapmap(model_statis: list, target):
    model_type = {}
    model_names = []
    model_predicts = []
    df = pd.DataFrame()
    for i in model_statis:
        weight_path = i[0]
        name, info = path_2_model_name(weight_path)
        if model_type.get(info[0]) is not None:
            continue
        model_type[info[0]] = 1
        predict = np.load(weight_path + ".predict.npy")
        model_predicts.append(predict)
        model_names.append(name)
        df[name] = predict.flatten()
    corr = df.corr()
    ax = sns.heatmap(corr, annot=True, cmap='YlGnBu')
    ax.set_xticklabels(ax.get_xticklabels(), rotation= 90)
    ax.set_yticklabels(ax.get_yticklabels(), rotation= 45)
    ax.get_figure().savefig(os.path.join(RECORD_DIR, target),dpi=100, bbox_inches='tight')


all_label, one_label = model_f2_statistics(path.MODEL_PATH, 1, "statistics_val1.txt")
model_f2_statistics(path.MODEL_PATH, 2, "statistics_val2.txt")
model_f2_statistics(path.MODEL_PATH, 3, "statistics_val3.txt")
model_f2_statistics(path.MODEL_PATH, 4, "statistics_val4.txt")
model_f2_statistics(path.MODEL_PATH, 5, "statistics_val5.txt")

model_corr_heapmap(all_label, 'test.png')
