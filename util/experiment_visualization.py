import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os

from util import data_loader
from util import data_metric
from util import path

label_class_type_name = ["WearScene", "AreaStyle", "AgeRange", "Others"]
label_class_name = ["Sport", "Leisure", "OL/commuting", "JapaneseStyle", "KoreanStyle", "EuropeStyle", "EnglandStyle",
                    "Maid", "Debutante/Lady", "Simple", "Nature", "Street/Punk", "Nation"]

TEST_FILE = path.TEST_RESULT_TXT
FIG_SAVE_FILE = os.path.join(path.ROOT_PATH, 'util', 'Fig2PPT')

def Test_get_columns():
    columns = []
    column_label_class_type_name = []
    for i in range(4):
        for j in range(data_metric.label_class_type_interval[i] - (
                data_metric.label_class_type_interval[i - 1] if i != 0 else -1)):
            column_label_class_type_name.append(data_metric.label_class_type_name[i])
    columns.append(column_label_class_type_name)
    columns.append(data_metric.label_class_name)
    return columns

def Test_get_labels(file):
    labels = []
    for line in open(file):
        label = line.split(',')[1:]
        labels.append(list(map(int, label)))
    return labels

def Test_get_frame(file):
    data = Test_get_labels(file)
    data = np.array(data)
    df = pd.DataFrame(data, columns=Test_get_columns())
    return df


def Test_get_label_class_sum(file):
    df = Test_get_frame(file)
    return df.sum(axis=0)


def Test_get_label_class_type_sum(file):
    df = Test_get_frame(file)
    label_class_type_sum = []
    for classType in data_metric.label_class_type_name:
        label_class_type_sum.append(df[classType].sum().sum())
    return label_class_type_sum


def Test_show_label_class_bar(file):
    plt.figure(figsize=(15, 6))
    _x = Test_get_label_class_sum(file).tolist()
    _y = label_class_name
    for i in range(len(_y)):
        _y[i] += '(%d)' % _x[i]
    sns.barplot(y=_y, x=_x, orient='h')
    # plt.show()
    savePath = os.path.join(FIG_SAVE_FILE, 'Test_label_class_bar')
    plt.savefig(savePath)


def Test_show_label_class_type_bar(file):
    plt.figure(figsize=(8, 6))
    _x = label_class_type_name
    _y = Test_get_label_class_type_sum(file)
    for i in range(len(_x)):
        _x[i] += '(%d)' % _y[i]
    sns.barplot(x=_x, y=_y)
    # plt.show()
    savePath = os.path.join(FIG_SAVE_FILE, 'Test_label_class_style_bar')
    plt.savefig(savePath)



if __name__ == "__main__":
    Test_show_label_class_bar(TEST_FILE)
    Test_show_label_class_type_bar(TEST_FILE)