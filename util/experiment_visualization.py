import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.gridspec as gridspec

import os
import json

from util import data_loader
from util import data_metric
from util import path

label_class_type_name = ["WearScene", "AreaStyle", "AgeRange", "Others"]
label_class_name = ["Sport", "Leisure", "OL/commuting", "JapaneseStyle", "KoreanStyle", "EuropeStyle", "EnglandStyle",
                    "Maid", "Debutante/Lady", "Simple", "Nature", "Street/Punk", "Nation"]
dic_label = ["num", "avg", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]
model_list = ["densenet169", "inceptionresnetv2", "resnet50", "xception"]

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

    plt.figure(figsize=(20, 18))

    for i in range(13):
        ax = plt.subplot(4, 4, i + 1)
        plt.title('label' + str(i))
        plt.scatter(x=diff_frame['num'].values, y=diff_frame[str(i)].values, marker='.')
        # sns.lmplot(x='num', y=str(i), data=diff_frame)
        # sns.distplot(diff_frame['avg'].values, hist=False, rug=True)
        # plt.xlabel('model_label'+str(i))
        # plt.ylabel('f2-score_diff')
        # plt.savefig(os.path.join(FIG_SAVE_FILE, "epoch_cv_test_info_diff_label" + str(i)))
    # sns.lmplot(x='num', y='avg', data=diff_frame)
    # sns.distplot(diff_frame['avg'].values, hist=False, rug=True)
    plt.subplot(4, 4, 16)
    plt.scatter(x=diff_frame['num'].values, y=diff_frame['avg'].values, marker='.')
    plt.title('avg')

    # plt.show()
    plt.savefig(os.path.join(FIG_SAVE_FILE, "epoch_cv_test_info_diff"))

def get_epoch_cv_test_info_each_CNNS(file_cv=path.EPOCH_CV, file_test=path.EPOCH_TEST):
    dic_cv = data_from_json(file_cv)
    dic_test = data_from_json(file_test)
    dic_diff = {}
    model_dic = {}
    for k_cv in dic_cv:
        if k_cv in dic_test:
            model = k_cv.split('\\')[2]
            if model not in dic_diff: dic_diff[model] = {}
            dic_diff[model][k_cv] = {}
            dic_diff[model][k_cv]["avg"] = dic_test[k_cv]["avg"] - dic_cv[k_cv]["avg"]
            dic_diff[model][k_cv]["0"] = dic_test[k_cv]["0"] - dic_cv[k_cv]["0"]
            dic_diff[model][k_cv]["1"] = dic_test[k_cv]["1"] - dic_cv[k_cv]["1"]
            dic_diff[model][k_cv]["2"] = dic_test[k_cv]["2"] - dic_cv[k_cv]["2"]
            dic_diff[model][k_cv]["3"] = dic_test[k_cv]["3"] - dic_cv[k_cv]["3"]
            dic_diff[model][k_cv]["4"] = dic_test[k_cv]["4"] - dic_cv[k_cv]["4"]
            dic_diff[model][k_cv]["5"] = dic_test[k_cv]["5"] - dic_cv[k_cv]["5"]
            dic_diff[model][k_cv]["6"] = dic_test[k_cv]["6"] - dic_cv[k_cv]["6"]
            dic_diff[model][k_cv]["7"] = dic_test[k_cv]["7"] - dic_cv[k_cv]["7"]
            dic_diff[model][k_cv]["8"] = dic_test[k_cv]["8"] - dic_cv[k_cv]["8"]
            dic_diff[model][k_cv]["9"] = dic_test[k_cv]["9"] - dic_cv[k_cv]["9"]
            dic_diff[model][k_cv]["10"] = dic_test[k_cv]["10"] - dic_cv[k_cv]["10"]
            dic_diff[model][k_cv]["11"] = dic_test[k_cv]["11"] - dic_cv[k_cv]["11"]
            dic_diff[model][k_cv]["12"] = dic_test[k_cv]["12"] - dic_cv[k_cv]["12"]
    for mod in dic_diff:
        dic_list = []
        number = 0
        for k_diff in dic_diff[mod]:
            number += 1
            dic_list.append([number])
            for k, v in dic_diff[mod][k_diff].items():
                dic_list[-1].append(v)
        diff_array = np.array(dic_list)
        diff_frame = pd.DataFrame(diff_array, columns=dic_label)
        model_dic[mod] = diff_frame


    plt.figure(figsize=(20, 18))
    for i in range(13):
        ax = plt.subplot(4, 4, i + 1)
        plt.title('label' + str(i))
        # plt.scatter(x=diff_frame['num'].values, y=diff_frame[str(i)].values, marker='.')
        # sns.lmplot(x='num', y=str(i), data=diff_frame)
        for mod in model_list: sns.distplot(model_dic[mod][str(i)].values, hist=False, rug=False, ax=ax, label=mod)
        # plt.xlabel('model_label'+str(i))
        # plt.ylabel('f2-score_diff')
        # plt.savefig(os.path.join(FIG_SAVE_FILE, "epoch_cv_test_info_diff_label" + str(i)))
        # sns.lmplot(x='num', y='avg', data=diff_frame)
        # sns.distplot(diff_frame['avg'].values, hist=False, rug=True)
    ax = plt.subplot(4, 4, 16)
    plt.title('avg')
    # plt.scatter(x=diff_frame['num'].values, y=diff_frame['avg'].values, marker='.')
    for mod in model_list: sns.distplot(model_dic[mod]['avg'].values, hist=False, rug=False, ax=ax, label=mod)
    # plt.show()
    plt.savefig(os.path.join(FIG_SAVE_FILE, "epoch_cv_test_info_diff_model"))

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
    plt.figure(figsize=(20, 18))
    for i in range(13):
        # sns.lmplot(x='num', y=str(i), data=diff_frame)
        ax = plt.subplot(4, 4, i + 1)
        plt.title('label' + str(i))
        sns.distplot(diff_frame[str(i)].values, hist=False, rug=True, ax=ax)
        # plt.scatter(x=diff_frame['num'].values, y=diff_frame[str(i)].values, marker='.')
        # plt.xlabel('label'+str(i))
        # plt.savefig(os.path.join(FIG_SAVE_FILE, "threshold_cv_test_info_diff_label" + str(i)))
        # plt.clf()
    # plt.show()
    plt.savefig(os.path.join(FIG_SAVE_FILE, "threshold_cv_test_info_diff"))

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
    plt.figure(figsize=(20, 18))
    for i in range(13):
        ax = plt.subplot(4, 4, i+1)
        plt.title('label'+str(i))
        plt.scatter(x=diff_frame['num'].values, y=diff_frame[str(i)].values, marker='.')
        # sns.lmplot(x='num', y=str(i), data=diff_frame)
        # sns.distplot(diff_frame[str(i)].values, hist=False, rug=True, ax=ax)
        # plt.xlabel('model_label'+str(i))
        # plt.savefig(os.path.join(FIG_SAVE_FILE, "epoch_test_standard_info_diff_label" + str(i)))
        # plt.clf()
    # plt.subplots_adjust(hspace=0.3)
    plt.savefig(os.path.join(FIG_SAVE_FILE, "epoch_test_standard_info_diff"))

ensemble_list = ['2005', '2055', '2105', '2155', '2205']
ensemble_label_list = ['top-1', 'top-5', 'top-10', 'top-15', 'top-20']
def get_ensemble_cv_test_info(file_cv=path.ENSEMBLE_CV, file_test=path.ENSEMBLE_TEST):
    dic_cv = data_from_json(file_cv)
    dic_test = data_from_json(file_test)
    ensemble_cv_dic = {}
    ensemble_test_dic = {}
    ensemble_cv_test_diff_dic = {}
    ensemble_test_avg_xgb_dic = {}
    ensemble_test_avg_cnn_dic = {}
    x_label = dic_label[1:]

    best_single_model_test_dic = data_from_json(path.GLOBAL_TEST)
    best_single_model_cv_dic = data_from_json(path.GLOBAL_CV)
    best_single_model_test_list = []
    best_single_model_cv_list = []
    for x in x_label:
        best_single_model_cv_list.append(best_single_model_cv_dic[x])
        best_single_model_test_list.append(best_single_model_test_dic[x])
    plt.figure(figsize=(15, 15))

    for xgb in ensemble_list:
        ensemble_cv_dic[xgb] = []
        ensemble_test_dic[xgb] = []
        ensemble_cv_test_diff_dic[xgb] = []
        ensemble_test_avg_xgb_dic[xgb] = []
        ensemble_test_avg_cnn_dic[xgb] = []
        for x in x_label:
            ensemble_cv_dic[xgb].append(dic_cv['xgb_model'+xgb][x])
            ensemble_test_dic[xgb].append(dic_test['xgb_model'+xgb+'.txt'][x])
            ensemble_cv_test_diff_dic[xgb].append(dic_test['xgb_model' + xgb + '.txt'][x] - dic_cv['xgb_model'+xgb][x])
            ensemble_test_avg_xgb_dic[xgb].append(dic_test['xgb_model' + xgb + '_avg[xgb].txt'][x])
            ensemble_test_avg_cnn_dic[xgb].append(dic_test['xgb_model' + xgb + '_avg[cnn].txt'][x])
    plt.subplot(221)
    for xgb in ensemble_list:
        plt.plot(x_label, ensemble_cv_dic[xgb], 'o--')
    plt.plot(x_label, best_single_model_cv_list, 'o--')
    plt.legend(ensemble_label_list+['SINGLE_MODEL'])
    plt.title('CV')
    plt.ylabel('f2-score')

    plt.subplot(222)
    for xgb in ensemble_list:
        plt.plot(x_label, ensemble_test_dic[xgb], 'o--')
    plt.plot(x_label, best_single_model_test_list, 'o--')
    plt.legend(ensemble_label_list + ['SINGLE_MODEL'])
    plt.title('TEST')
    plt.ylabel('f2-score')

    plt.subplot(223)
    for xgb in ensemble_list:
        plt.plot(x_label, ensemble_cv_test_diff_dic[xgb], 'o--')
    plt.legend(ensemble_label_list)
    plt.title('TEST - CV')
    plt.ylabel('f2-score_diff')

    plt.subplot(224)
    # for xgb in ensemble_list:
    for xgb in ['2055']:
        plt.plot(x_label, ensemble_test_dic[xgb], 'o--')
        plt.plot(x_label, ensemble_test_avg_xgb_dic[xgb], 'o--')
        plt.plot(x_label, ensemble_test_avg_cnn_dic[xgb], 'o--')
    plt.legend(['vote', 'avg[xgb]', 'avg[cnn]'])
    plt.title('multi-level ensemble')
    plt.ylabel('f2-score')
    plt.savefig(os.path.join(FIG_SAVE_FILE, "ensemble_cv_test_info_top"))

#研究多级集成与top的关系
    vote_label0 = []
    avg_xgb_label0 = []
    avg_cnn_label0 = []
    plt.figure(figsize=(20, 15))
    for xgb in ensemble_list:
        vote_label0.append(ensemble_test_dic[xgb][1])
        avg_xgb_label0.append(ensemble_test_avg_xgb_dic[xgb][1])
        avg_cnn_label0.append(ensemble_test_avg_cnn_dic[xgb][1])
        n = ensemble_list.index(xgb)
        plt.subplot(2, 3, n+1)
        plt.plot(x_label, ensemble_test_dic[xgb], 'o--')
        plt.plot(x_label, ensemble_test_avg_xgb_dic[xgb], 'o--')
        plt.plot(x_label, ensemble_test_avg_cnn_dic[xgb], 'o--')
        plt.legend(['vote', 'avg[xgb]', 'avg[cnn]'])
        plt.title(ensemble_label_list[n])
        plt.ylabel('f2-score')

    plt.subplot(2, 3, 6)
    plt.plot(ensemble_label_list, vote_label0)
    plt.plot(ensemble_label_list, avg_xgb_label0)
    plt.plot(ensemble_label_list, avg_cnn_label0)
    plt.legend(['vote', 'avg[xgb]', 'avg[cnn]'])
    plt.ylabel('f2-score')
    plt.title('multi-ensemble label0')
    plt.savefig(os.path.join(FIG_SAVE_FILE, "ensemble_cv_test_info_top_multi_ensemble"))

#最好单模型标签和最好集成标签比较
    plt.figure(figsize=(15, 8))
    plt.subplot(121)
    plt.plot(x_label, best_single_model_test_list, 'o--')
    plt.plot(x_label, ensemble_test_avg_cnn_dic['2155'], 'o--')
    plt.legend(['BEST_SINGLE_MODEL', 'BEST_MULTI_ENSEMBLE'])
    plt.ylabel('f2-score')
    plt.subplot(122)
    plt.plot(x_label, [i - j for i, j in zip(ensemble_test_avg_cnn_dic['2155'], best_single_model_test_list)], 'o--')
    plt.ylabel('f2-score_diff')
    plt.savefig(os.path.join(FIG_SAVE_FILE, "ensemble_cv_test_info_bestSingleModel_vs_bestMultiEnsemble"))

if __name__ == "__main__":
    # Test_show_label_class_bar(TEST_FILE)
    # Test_show_label_class_type_bar(TEST_FILE)
    # get_epoch_cv_test_info()
    # get_global_cv_test_info()
    # get_threshold_cv_test_info()
    # get_epoch_test_standard_info()
    # get_epoch_cv_test_info_each_CNNS()
    # get_ensemble_cv_test_info()
    pass