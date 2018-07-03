# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from util import data_loader
from util import data_metric
from util import path

label_class_type_name = ["WearScene", "AreaStyle", "AgeRange", "Others"]
label_class_name = ["Sport", "Leisure", "OL/commuting", "JapaneseStyle", "KoreanStyle", "EuropeStyle", "EnglandStyle",
                    "Maid", "Debutante/Lady", "Simple", "Nature", "Street/Punk", "Nation"]

DATA_FILE = path.ORIGINAL_TRAIN_IMAGES_PATH
DATA_COLOR_FILE = path.COLOR_AUGMENTED_PATH


def get_columns():
    columns = []
    column_label_class_type_name = []
    for i in range(4):
        for j in range(data_metric.label_class_type_interval[i] - (
                data_metric.label_class_type_interval[i - 1] if i != 0 else -1)):
            column_label_class_type_name.append(data_metric.label_class_type_name[i])
    columns.append(column_label_class_type_name)
    columns.append(data_metric.label_class_name)
    return columns


def get_frame(file):
    data = data_loader.get_labels(data_loader.list_image_name(file))
    data = np.array(data)
    df = pd.DataFrame(data, columns=get_columns())
    return df


def get_label_class_sum():
    df = get_frame(DATA_FILE)
    return df.sum(axis=0)


def get_label_class_type_sum():
    df = get_frame(DATA_FILE)
    label_class_type_sum = []
    for classType in data_metric.label_class_type_name:
        label_class_type_sum.append(df[classType].sum().sum())
    return label_class_type_sum


def show_label_class_bar():
    plt.figure(1, figsize=(15, 6))
    _x = get_label_class_sum().tolist()
    _y = label_class_name
    for i in range(len(_y)):
        _y[i] += '(%d)' % _x[i]
    sns.barplot(y=_y, x=_x, orient='h')
    plt.show()


def show_label_class_type_bar():
    plt.figure(1, figsize=(6, 6))
    sns.barplot(x=label_class_type_name, y=get_label_class_type_sum())
    plt.show()


def get_label_dic():
    data_list1 = data_loader.get_labels(data_loader.list_image_name(DATA_FILE))
    data_list2 = data_loader.get_labels(data_loader.list_image_name(DATA_COLOR_FILE))
    data_list = data_list1 + data_list2
    data_dic = {}
    for _data in data_list:
        data_ = str(_data)
        if data_ not in list(data_dic):
            data_dic[data_] = 1
        else:
            data_dic[data_] += 1
    with open("data_dic.txt", 'w') as f:
        for k, v in data_dic.items():
            log = k + " : " + str(v) + '\n'
            f.write(log)


def show_label_calss_bar_per_epoch(train_file, record_dir):
    labels = data_loader.get_labels(train_file)
    labels = np.array(labels)
    df = pd.DataFrame(labels, columns=get_columns())
    _x = df.sum(axis=0).tolist()
    _y = label_class_name.copy()
    for i in range(len(_y)):
        _y[i] += '(%d)' % _x[i]
    plt.figure(1, figsize=(15, 6))
    sns.barplot(y=_y, x=_x, orient='h')
    record_save = record_dir + '_' + 'label_calss_bar'
    plt.savefig(record_save)


if __name__ == "__main__":
    # show_label_class_bar()
    # show_label_class_type_bar()
    get_label_dic()
