import os
import pathlib
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import fbeta_score

import config
from util import data_loader
from util import keras_util
from util import path

RECORD_DIR = os.path.join(os.path.abspath("."), "record")


def model_f2_statistics_no_repeat(all_label, one_label, thresholds, save_file):
    all_label_best = {}
    one_label_best = [{} for i in range(13)]
    one_label_no_repeat = {}
    all_label_no_repeat = []
    for entry in all_label:
        if entry[0].split('.')[0] not in all_label_best:
            all_label_best[entry[0].split('.')[0]] = entry[1]
            all_label_no_repeat.append(entry)

    for label in one_label:
        for entry in one_label[label]:
            if entry[0].split('.')[0] not in one_label_best[label]:
                one_label_best[label][entry[0].split('.')[0]] = entry[1]
                if one_label_no_repeat.get(label) is None:
                    one_label_no_repeat[label] = []
                one_label_no_repeat[label].append(entry)

    if save_file is not None:
        record_dir = os.path.join(RECORD_DIR, "val%d" % val_index)
        pathlib.Path(record_dir).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(record_dir, save_file), "w") as f:
            f.write("==========================All label==========================\n")
            for i in all_label_no_repeat:
                f.write("%f: %s\n" % (i[1], i[0]))

            for i in range(len(one_label_no_repeat)):
                f.write("\n\n\n\n\n==========================One label: %d==========================\n" % i)
                for j in one_label_no_repeat[i]:
                    f.write("%f: %s\n" % (j[1], j[0]))

    return all_label_no_repeat, one_label_no_repeat, thresholds


def model_f2_statistics(mode_path, val_index=1, save_file=None):
    """
    对model目录下的所有包含"evaluate"字段的文件进行统计，分别得到all-label、one-label统计
    :param mode_path: 需要统计的目录
    :param save_file: 输入文件
    :return:
    """
    evaluate_files = []
    for root, dirs, files in os.walk(mode_path):
        for file in files:
            if ("val%d" % val_index) not in root:
                continue
            if "evaluate" in file and "evaluate_revise" not in file:
                evaluate_files.append(os.path.join(root, file))

    all_label = {}
    one_label = {}
    label_f2_threshold = [{} for i in range(13)]
    for file in evaluate_files:
        with open(file, "r") as f:
            weight_file = ""
            for i in f.readlines():
                if "Weight" in i:
                    # 不同人训练出来的模型中，weight_file的根路径不同，此处进行一个转换
                    weight_file = os.path.join(path.root_path, re.match(r"Weight:.*competition\\*(.*)", i).group(1))
                if "Greedy F2-Score is:" in i:
                    if weight_file == "":
                        print("file %s is abnormal" % file)
                    greedy_f2 = i.split(":")[-1].strip()
                    all_label[weight_file] = float(greedy_f2)
                if "[label" in i:
                    if weight_file == "":
                        print("file %s is abnormal" % file)
                    label = re.match(r".*label *([0-9]*)", i).group(1)
                    greedy_f2 = re.match(r".*greedy-f2=(.*)\[", i).group(1)
                    threshold = re.match(r".*greedy-f2=.*\[(.*)\]", i).group(1)
                    if one_label.get(int(label), None) is None:
                        one_label[int(label)] = {}
                    one_label[int(label)][weight_file] = float(greedy_f2)
                    label_f2_threshold[int(label)][weight_file] = float(threshold)

    all_label = sorted(all_label.items(), key=lambda x: x[1], reverse=True)
    for i in range(len(one_label)):
        one_label[i] = sorted(one_label[i].items(), key=lambda x: x[1], reverse=True)

    if save_file is not None:
        record_dir = os.path.join(RECORD_DIR, "val%d" % val_index)
        pathlib.Path(record_dir).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(record_dir, save_file), "w") as f:
            f.write("==========================All label==========================\n")
            for i in all_label:
                f.write("%f: %s\n" % (i[1], i[0]))

            for i in range(len(one_label)):
                f.write("\n\n\n\n\n==========================One label: %d==========================\n" % i)
                for j in one_label[i]:
                    f.write("%f: %s\n" % (j[1], j[0]))

    return all_label, one_label, label_f2_threshold


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


def model_corr_heapmap(model_statis: list, label, thresholds, val_index, target):
    _, val_files = data_loader.get_k_fold_files("1.txt", val_index, [config.DATA_TYPE_ORIGINAL])
    y = data_loader.get_labels(val_files)
    y = np.array(y, np.int8).reshape((-1, 13))
    model_names = []
    model_predicts = []
    df = pd.DataFrame()
    for i in model_statis:
        weight_path = i[0]
        name, info = path_2_model_name(weight_path)
        if not os.path.exists(weight_path + ".predict.npy"):
            continue
        predict = np.load(weight_path + ".predict.npy")
        if thresholds is not None:
            for l in range(len(thresholds)):
                threshold = thresholds[l][weight_path]
                predict[:, l] = predict[:, l] > threshold

        predict = predict.astype(np.int8)

        if label is not None:
            predict = predict[:, label]
            f2 = fbeta_score(y[:, label], predict, beta=2)
        else:
            f2 = fbeta_score(y, predict, beta=2, average='macro')

        assert (f2 - i[1]) / i[1] < 0.01

        model_predicts.append(predict)
        model_names.append(name)
        df[name] = predict.flatten()
    corr = df.corr()
    plt.gcf().clear()
    ax = sns.heatmap(corr, annot=len(model_statis) < 10, cmap='YlGnBu')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    record_dir = os.path.join(RECORD_DIR, "val%d" % val_index)
    pathlib.Path(record_dir).mkdir(parents=True, exist_ok=True)
    ax.get_figure().savefig(os.path.join(record_dir, target), dpi=100, bbox_inches='tight')


def shord_board_statistics(label_statis_all):
    shord_board_statis = [[] for i in range(5)]
    for val in range(5):
        label_statis_val = label_statis_all[val]
        for label in range(13):
            label_statis = label_statis_val[label]
            average = 0
            for i in range(5):
                average += label_statis[i][1] / 5
            shord_board_statis[val].append(average)

    with open(os.path.join(RECORD_DIR, "short_board_statistics.txt"), 'w+') as f:
        f.write("Top-5 f2-score average\n")
        for i in range(13):
            f.write("\n#######label %d\n" % i)
            for j in range(5):
                f.write("val %d: %f\n" % (j + 1, shord_board_statis[j][i]))


def model_config_statistics(label_statis_all):
    with open(os.path.join(RECORD_DIR, "model_config_statistics.txt"), "w+") as f:
        for label in range(13):
            f.write("##############################label %d##############################\n" % label)
            for val in range(5):
                f.write("---------------------val %d---------------------\n" % (val + 1))
                for rank in range(2):
                    weight_file, f2 = label_statis_all[val][label][rank]
                    _, model_config = keras_util.dynamic_model_import(weight_file)
                    assert model_config.val_index == val + 1
                    f.write("rank: %d, f2-score: %6f\n" % (rank, f2))
                    f.write("model_name=%s\n" % model_config.model_name)
                    f.write("image_resolution=%d\n" % model_config.image_resolution)
                    f.write("data_type=%s\n" % str(model_config.data_type))
                    f.write("label_position=%s\n" % str([str(i) for i in model_config.label_position]))

                    f.write("train_file_cnt=%d\n" % model_config.train_file_cnt)
                    f.write("val_file_cnt=%d\n" % model_config.val_file_cnt)
                    try:
                        f.write("label_color_augment=%s\n" % str([str(i) for i in model_config.label_color_augment]))
                        f.write("color_augment_cnt=%d\n" % model_config.color_augment_cnt)
                    except:
                        pass

                    try:
                        f.write("label_up_sampling=%s\n" % str([str(i) for i in model_config.label_up_sampling]))
                        f.write("label_up_sampling_cnt=%s\n" % str([str(i) for i in model_config.up_sampling_cnt]))
                    except:
                        pass

                    try:
                        f.write("down_sampling=%f\n" % model_config.downsampling)
                        f.write("down_sampling_cnt=%d\n" % model_config.down_sampling_cnt)
                    except:
                        pass

                    f.write("train_batch_size=%s\n" % str([str(i) for i in model_config.train_batch_size]))
                    f.write("epoch=%s\n" % str([str(i) for i in model_config.epoch]))
                    f.write("lr=%s\n" % str([str(i) for i in model_config.lr]))
                    f.write("freeze_layers=%s\n" % str([str(i) for i in model_config.freeze_layers]))
                    f.write("input_norm=%s\n" % model_config.input_norm)
                    f.write("tta_flip=%s\n" % model_config.tta_flip)
                    f.write("tta_crop=%s\n" % model_config.tta_crop)
                    f.write("\n")

one_label_all = []
for val_index in range(1, 6):
    all_label, one_label, thresholds = model_f2_statistics(path.MODEL_PATH, val_index,
                                                           "statistics_val%d_all.txt" % val_index)
    all_label, one_label, thresholds = model_f2_statistics_no_repeat(all_label, one_label, thresholds,
                                                                     "statistics_val%d_no_repeat.txt" % val_index)
    one_label_all.append(one_label)
    #
    # model_corr_heapmap(all_label[:20], None, thresholds, val_index, "label_all.png")
    # for i in range(13):
    #     model_corr_heapmap(one_label[i][:20], i, thresholds, val_index, 'label_%d.png' % i)

shord_board_statistics(one_label_all)
# model_config_statistics(one_label_all)
