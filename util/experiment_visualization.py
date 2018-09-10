import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import json

from util import data_loader
from util import data_metric
from util import path

label_class_type_name = ["WearScene", "AreaStyle", "AgeRange", "Others"]
label_class_name = ["Sport", "Leisure", "OL/commuting", "JapaneseStyle", "KoreanStyle", "EuropeStyle", "EnglandStyle",
                    "Maid", "Debutante/Lady", "Simple", "Nature", "Street/Punk", "Nation"]
dic_label = ["num", "avg", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]

TEST_FILE = path.TEST_RESULT_TXT
FIG_SAVE_FILE = os.path.join(path.ROOT_PATH, 'util', 'Fig2PPT')


sns.set_style("whitegrid")

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

def data_from_json(file):
    with open(file) as f:
        data = json.load(f)
        return data

def get_epoch_cv_test_info(file_cv=path.EPOCH_CV, file_test=path.EPOCH_TEST):
    dic_cv = data_from_json(file_cv)
    dic_test = data_from_json(file_test)
    dic_diff = {}
    dic_list = []
    for k_cv in dic_cv:
        if k_cv in dic_test:
            dic_diff[k_cv] = {}
            dic_diff[k_cv]["avg"] = dic_test[k_cv]["avg"] - dic_cv[k_cv]["avg"]
            dic_diff[k_cv]["0"] = dic_test[k_cv]["0"] - dic_cv[k_cv]["0"]
            dic_diff[k_cv]["1"] = dic_test[k_cv]["1"] - dic_cv[k_cv]["1"]
            dic_diff[k_cv]["2"] = dic_test[k_cv]["2"] - dic_cv[k_cv]["2"]
            dic_diff[k_cv]["3"] = dic_test[k_cv]["3"] - dic_cv[k_cv]["3"]
            dic_diff[k_cv]["4"] = dic_test[k_cv]["4"] - dic_cv[k_cv]["4"]
            dic_diff[k_cv]["5"] = dic_test[k_cv]["5"] - dic_cv[k_cv]["5"]
            dic_diff[k_cv]["6"] = dic_test[k_cv]["6"] - dic_cv[k_cv]["6"]
            dic_diff[k_cv]["7"] = dic_test[k_cv]["7"] - dic_cv[k_cv]["7"]
            dic_diff[k_cv]["8"] = dic_test[k_cv]["8"] - dic_cv[k_cv]["8"]
            dic_diff[k_cv]["9"] = dic_test[k_cv]["9"] - dic_cv[k_cv]["9"]
            dic_diff[k_cv]["10"] = dic_test[k_cv]["10"] - dic_cv[k_cv]["10"]
            dic_diff[k_cv]["11"] = dic_test[k_cv]["11"] - dic_cv[k_cv]["11"]
            dic_diff[k_cv]["12"] = dic_test[k_cv]["12"] - dic_cv[k_cv]["12"]
    number = 0
    for k_diff in dic_diff:
        number += 1
        dic_list.append([number])
        for k, v in dic_diff[k_diff].items():
            dic_list[-1].append(v)
    diff_array = np.array(dic_list)
    diff_frame = pd.DataFrame(diff_array, columns=dic_label)

    sns.lmplot(x='num', y='avg', data=diff_frame)
    # sns.distplot(diff_frame['avg'].values, hist=False, rug=True)
    plt.xlabel('model_avg')
    plt.ylabel('f2-score_diff')
    plt.savefig(os.path.join(FIG_SAVE_FILE, "epoch_cv_test_info_avg"))

    for i in range(13):
        sns.lmplot(x='num', y=str(i), data=diff_frame)
        # sns.distplot(diff_frame['avg'].values, hist=False, rug=True)
        plt.xlabel('model_label'+str(i))
        plt.ylabel('f2-score_diff')
        plt.savefig(os.path.join(FIG_SAVE_FILE, "epoch_cv_test_info_diff_label" + str(i)))

def get_global_cv_test_info(file_cv=path.GLOBAL_CV, file_test=path.GLOBAL_TEST):
    dic_cv = data_from_json(file_cv)
    dic_test = data_from_json(file_test)
    list_cv = []
    list_test = []
    x_label = dic_label[1:]
    for x in x_label:
        list_cv.append(dic_cv[x])
        list_test.append(dic_test[x])
    plt.plot(x_label, list_cv, 'o--')
    plt.plot(x_label, list_test, 'o--')
    plt.legend(['CV', 'TEST'])
    plt.ylabel('f2-score')
    plt.savefig(os.path.join(FIG_SAVE_FILE, "global_cv_test_info"))

def get_threshold_cv_test_info(file_cv=path.THRESHOLD_CV, file_test=path.THRESHOLD_TEST):
    dic_cv = data_from_json(file_cv)
    dic_test = data_from_json(file_test)
    dic_diff = {}
    dic_list = []
    for k_cv in dic_cv:
        if k_cv in dic_test:
            dic_diff[k_cv] = {}
            dic_diff[k_cv]["0"] = dic_test[k_cv]["0"] - dic_cv[k_cv]["0"]
            dic_diff[k_cv]["1"] = dic_test[k_cv]["1"] - dic_cv[k_cv]["1"]
            dic_diff[k_cv]["2"] = dic_test[k_cv]["2"] - dic_cv[k_cv]["2"]
            dic_diff[k_cv]["3"] = dic_test[k_cv]["3"] - dic_cv[k_cv]["3"]
            dic_diff[k_cv]["4"] = dic_test[k_cv]["4"] - dic_cv[k_cv]["4"]
            dic_diff[k_cv]["5"] = dic_test[k_cv]["5"] - dic_cv[k_cv]["5"]
            dic_diff[k_cv]["6"] = dic_test[k_cv]["6"] - dic_cv[k_cv]["6"]
            dic_diff[k_cv]["7"] = dic_test[k_cv]["7"] - dic_cv[k_cv]["7"]
            dic_diff[k_cv]["8"] = dic_test[k_cv]["8"] - dic_cv[k_cv]["8"]
            dic_diff[k_cv]["9"] = dic_test[k_cv]["9"] - dic_cv[k_cv]["9"]
            dic_diff[k_cv]["10"] = dic_test[k_cv]["10"] - dic_cv[k_cv]["10"]
            dic_diff[k_cv]["11"] = dic_test[k_cv]["11"] - dic_cv[k_cv]["11"]
            dic_diff[k_cv]["12"] = dic_test[k_cv]["12"] - dic_cv[k_cv]["12"]
    number = 0
    for k_diff in dic_diff:
        number += 1
        dic_list.append([number])
        for k, v in dic_diff[k_diff].items():
            dic_list[-1].append(v)
    diff_array = np.array(dic_list)
    diff_frame = pd.DataFrame(diff_array, columns=dic_label[:1]+dic_label[2:])
    for i in range(13):
        # sns.lmplot(x='num', y=str(i), data=diff_frame)
        sns.distplot(diff_frame[str(i)].values, hist=False, rug=True)
        plt.xlabel('label'+str(i))
        plt.savefig(os.path.join(FIG_SAVE_FILE, "threshold_cv_test_info_diff_label" + str(i)))
        plt.clf()


def get_epoch_test_standard_info(file_cv=path.EPOCH_TEST, file_test=path.EPOCH_TEST_STANDARD):
    dic_cv = data_from_json(file_cv)
    dic_test = data_from_json(file_test)
    dic_diff = {}
    dic_list = []
    for k_cv in dic_cv:
        if k_cv in dic_test:
            dic_diff[k_cv] = {}
            dic_diff[k_cv]["0"] = dic_test[k_cv]["0"] - dic_cv[k_cv]["0"]
            dic_diff[k_cv]["1"] = dic_test[k_cv]["1"] - dic_cv[k_cv]["1"]
            dic_diff[k_cv]["2"] = dic_test[k_cv]["2"] - dic_cv[k_cv]["2"]
            dic_diff[k_cv]["3"] = dic_test[k_cv]["3"] - dic_cv[k_cv]["3"]
            dic_diff[k_cv]["4"] = dic_test[k_cv]["4"] - dic_cv[k_cv]["4"]
            dic_diff[k_cv]["5"] = dic_test[k_cv]["5"] - dic_cv[k_cv]["5"]
            dic_diff[k_cv]["6"] = dic_test[k_cv]["6"] - dic_cv[k_cv]["6"]
            dic_diff[k_cv]["7"] = dic_test[k_cv]["7"] - dic_cv[k_cv]["7"]
            dic_diff[k_cv]["8"] = dic_test[k_cv]["8"] - dic_cv[k_cv]["8"]
            dic_diff[k_cv]["9"] = dic_test[k_cv]["9"] - dic_cv[k_cv]["9"]
            dic_diff[k_cv]["10"] = dic_test[k_cv]["10"] - dic_cv[k_cv]["10"]
            dic_diff[k_cv]["11"] = dic_test[k_cv]["11"] - dic_cv[k_cv]["11"]
            dic_diff[k_cv]["12"] = dic_test[k_cv]["12"] - dic_cv[k_cv]["12"]
    number = 0
    for k_diff in dic_diff:
        number += 1
        dic_list.append([number])
        for k, v in dic_diff[k_diff].items():
            dic_list[-1].append(v)
    diff_array = np.array(dic_list)
    diff_frame = pd.DataFrame(diff_array, columns=dic_label[:1] + dic_label[2:])
    for i in range(13):
        sns.lmplot(x='num', y=str(i), data=diff_frame)
        # sns.distplot(diff_frame[str(i)].values, hist=False, rug=True)
        plt.xlabel('model_label'+str(i))
        plt.ylabel('f2-score_diff')
        plt.savefig(os.path.join(FIG_SAVE_FILE, "epoch_test_standard_info_diff_label" + str(i)))
        plt.clf()

if __name__ == "__main__":
    # Test_show_label_class_bar(TEST_FILE)
    # Test_show_label_class_type_bar(TEST_FILE)
    # get_epoch_cv_test_info()
    # get_global_cv_test_info()
    # get_threshold_cv_test_info()
    # get_epoch_test_standard_info()
    pass