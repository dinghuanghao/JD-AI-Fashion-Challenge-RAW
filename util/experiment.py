import json
import os
import re

import numpy as np
from sklearn.metrics import fbeta_score

from util import metrics
from util import keras_util
from util import model_statistics
from util import path


def get_epoch_cv():
    with open(path.EPOCH_CV, "r") as f:
        return json.load(f)


def get_epoch_test():
    with open(path.EPOCH_TEST, "r") as f:
        return json.load(f)


def get_epoch_test_standard():
    with open(path.EPOCH_TEST_STANDARD, "r") as f:
        return json.load(f)


def get_model_cv():
    with open(path.MODEL_CV, "r") as f:
        return json.load(f)


def get_model_test():
    with open(path.MODEL_TEST, "r") as f:
        return json.load(f)


def get_global_cv():
    with open(path.GLOBAL_CV, "r") as f:
        return json.load(f)


def get_threshold_cv():
    with open(path.THRESHOLD_CV, "r") as f:
        return json.load(f)


def get_threshold_test():
    with open(path.THRESHOLD_TEST, "r") as f:
        return json.load(f)


def get_global_test():
    with open(path.GLOBAL_TEST, "r") as f:
        return json.load(f)


def save_epoch_cv(dic):
    with open(path.EPOCH_CV, "w+") as f:
        json.dump(dic, f)


def save_epcoh_test(dic):
    with open(path.EPOCH_TEST, "w+") as f:
        json.dump(dic, f)


def save_epoch_test_standard(dic):
    with open(path.EPOCH_TEST_STANDARD, "w+") as f:
        json.dump(dic, f)


def save_model_cv(dic):
    with open(path.MODEL_CV, "w+") as f:
        json.dump(dic, f)


def save_model_test(dic):
    with open(path.MODEL_TEST, "w+") as f:
        json.dump(dic, f)


def save_global_cv(dic):
    with open(path.GLOBAL_CV, "w+") as f:
        json.dump(dic, f)


def save_global_test(dic):
    with open(path.GLOBAL_TEST, "w+") as f:
        json.dump(dic, f)


def save_threshold_cv(dic):
    with open(path.THRESHOLD_CV, "w+") as f:
        json.dump(dic, f)


def save_threshold_test(dic):
    with open(path.THRESHOLD_TEST, "w+") as f:
        json.dump(dic, f)


def get_epoch_identifier(weight_path):
    return re.search(r".*competition(.*)", weight_path).group(1)


def get_model_identifier(weight_path):
    weight_path = weight_path.split(".")[0]
    return re.search(r".*competition(.*)", weight_path).group(1)


def get_xgb_result():
    labels = []
    with open(os.path.join(path.RESULT_PATH, "xgb_model1205_avg[xgb].txt"), "r") as f:
        for i in f.readlines():
            result = i.strip().split(",")[1:]
            result = [int(c) for c in result]
            labels.append(result)

    return np.array(labels, np.int8)


def get_test_labels():
    labels = []
    with open(path.TEST_RESULT_TXT, "r") as f:
        for i in f.readlines():
            result = i.strip().split(",")[1:]
            result = [int(c) for c in result]
            labels.append(result)

    return np.array(labels, np.int8)


def build_epoch_test(y_true, y_pred, weight_file, identifier=None):
    all_f2 = get_epoch_test()
    if not identifier:
        identifier = get_epoch_identifier(weight_file)

    if all_f2.get(identifier):
        print(f"prediction is evaluated, {identifier}")
        return
    one_label_greedy_f2_all = []
    for i in range(13):
        one_label_greedy_f2_all.append(fbeta_score(y_true[:, i], y_pred[:, i], beta=2))

    f2 = {"avg": np.mean(one_label_greedy_f2_all)}
    for i in range(13):
        f2[f"{i}"] = one_label_greedy_f2_all[i]

    all_f2[identifier] = f2
    save_epcoh_test(all_f2)


def get_ablation_experiment_predict(mode_path, val):
    y_true = get_test_labels()

    original_test_file = []
    with open(path.TEST_DATA_TXT, 'r') as f:
        for i in f.readlines():
            image_name = i.split(",")[0] + ".jpg"
            original_test_file.append(os.path.join(path.ORIGINAL_TEST_IMAGES_PATH, image_name))

    weight_files = []
    predict_files = []
    _, _, thresholds = model_statistics.model_f2_statistics(path.MODEL_PATH, val)

    for root, dirs, files in os.walk(mode_path):
        for file in files:
            if "hdf5(test)" in file:
                predict_files.append(os.path.join(root, file))
                continue
            if not file.split(".")[-1] == "hdf5":
                continue
            if f"val{val}" not in root:
                continue

            model_num = re.match(r".*model([0-9]*).*", root).group(1)
            if int(model_num) < 100:
                continue

            weight_files.append(os.path.join(root, file))

    for predict_file in predict_files:
        print(f"evaluate {predict_file}")
        weight_file = predict_file.replace("(test).predict.npy", "")
        y_pred = np.load(predict_file)
        build_epoch_test(y_true, y_pred, weight_file)

    for weight_file in weight_files:
        print(f"evaluate {weight_file}")
        unique_path = re.match(r".*competition[\\/]*(.*)", weight_file).group(1)
        identifier = "-".join(unique_path.split("\\"))
        print(f"id {identifier}")
        cnn_result_path = os.path.join(weight_file + "(test)")
        print(f"result {cnn_result_path}")
        if not os.path.exists(keras_util.get_prediction_path(cnn_result_path)):
            print(cnn_result_path)
            attr_get_model, attr_model_config = keras_util.dynamic_model_import(weight_file)
            model = attr_get_model(output_dim=len(attr_model_config.label_position), weights=None)
            model.load_weights(weight_file)
            attr_model_config.val_files = []
            for data_type in attr_model_config.data_type:
                if data_type == path.DATA_TYPE_ORIGINAL:
                    attr_model_config.val_files.append(original_test_file)

            attr_model_config.tta_flip = True
            y_pred = keras_util.predict_tta(model, attr_model_config, verbose=1)

            for i in range(13):
                y_pred[:, i] = y_pred[:, i] > thresholds[i][weight_file]

            y_pred = y_pred.astype(np.int8)

            keras_util.save_prediction_file(y_pred, cnn_result_path)
        else:
            y_pred = np.load(keras_util.get_prediction_path(cnn_result_path))

        build_epoch_test(y_true, y_pred, weight_file)


def calc_xgb_f2_score():
    y_true = get_test_labels()
    y_pred = get_xgb_result()

    with open(os.path.join(path.RESULT_PATH, "test.txt"), "a+") as f:
        one_label_greedy_f2_all = []

        for i in range(13):
            one_label_greedy_f2_all.append(fbeta_score(y_true[:, i], y_pred[:, i], beta=2))

        f.write("\n\n")
        f.write("Greedy F2-Score is: %f\n" % np.average(one_label_greedy_f2_all))
        for i in range(13):
            f.write("[label %d] greedy-f2=%4f\n" % (i, one_label_greedy_f2_all[i]))


def get_existed_cnn_f2_score(val, mode_path):
    y_true = get_test_labels()

    original_test_file = []
    cnt = 0
    with open(path.TEST_DATA_TXT, 'r') as f:
        for i in f.readlines():
            image_name = i.split(",")[0] + ".jpg"
            original_test_file.append(os.path.join(path.ORIGINAL_TEST_IMAGES_PATH, image_name))

    weight_files = []
    _, _, thresholds = model_statistics.model_f2_statistics(path.MODEL_PATH, val)

    for root, dirs, files in os.walk(mode_path):
        for file in files:
            if "predict.np" not in file or f"val{val}" not in root:
                continue
            weight_files.append(os.path.join(root, file)[:-12])

    result = [{} for i in range(13)]
    average = {}

    with open(os.path.join(path.RESULT_PATH, f"test(training)(all)(val{val}).txt"), "w+") as f:
        for weight_file in weight_files:
            print(f"weight file {weight_file}")
            unique_path = re.match(r".*competition[\\/]*(.*)", weight_file).group(1)
            identifier = "-".join(unique_path.split("\\"))
            print(f"id {identifier}")
            cnn_result_path = os.path.join(path.CNN_RESULT_PATH, identifier)
            print(f"result {cnn_result_path}")
            if os.path.exists(keras_util.get_prediction_path(cnn_result_path)):
                y_pred = np.load(keras_util.get_prediction_path(cnn_result_path))
            else:
                print("model not existed")
                continue

            if thresholds[0].get(weight_file) is None:
                print("threshold not existed")
                continue

            for i in range(13):
                y_pred[:, i] = y_pred[:, i] > thresholds[i][weight_file]

            y_pred = y_pred.astype(np.int8)

            one_label_greedy_f2_all = []

            for i in range(13):
                # f2, _  = metrics.greedy_f2_score(y_true[:, i], y_pred[:, i], 1)
                f2 = fbeta_score(y_true[:, i], y_pred[:, i], beta=2)
                one_label_greedy_f2_all.append(f2)
                result[i][weight_file] = f2
            average[weight_file] = np.mean(one_label_greedy_f2_all)

            f.write("\n\n")
            f.write("Weight: %s\n" % cnn_result_path)
            f.write("Greedy F2-Score is: %f\n" % np.mean(one_label_greedy_f2_all))
            for i in range(13):
                f.write("[label %d] greedy-f2=%4f\n" % (i, one_label_greedy_f2_all[i]))

            print(f"need predict {cnt} model")

    with open(os.path.join(path.RESULT_PATH, f"test(training)(label)(val{val}).txt"), "w+") as f:
        average = sorted(average.items(), key=lambda x: x[1], reverse=True)
        f.write(f"\n\n==================== all =======================\n\n")
        for item in average:
            f.write("%4f: %s \n" % (item[1], item[0]))

        for i in range(13):
            dic = result[i]
            dic = sorted(dic.items(), key=lambda x: x[1], reverse=True)
            f.write(f"\n\n====================label {i} =======================\n\n")
            for item in dic:
                f.write("%4f: %s \n" % (item[1], item[0]))


def build_epoch_cv(val):
    all_label, one_label, thresholds = model_statistics.model_f2_statistics(path.MODEL_PATH, val)
    all_f2 = get_epoch_cv()

    for all in all_label:
        f2 = {}
        f2["avg"] = all[1]
        all_f2[get_epoch_identifier(all[0])] = f2

    for label in range(13):
        for one in one_label[label]:
            identifier = get_epoch_identifier(one[0])
            all_f2[identifier][f"{label}"] = one[1]

    save_epoch_cv(all_f2)


def build_model_cv(val):
    all_label, one_label, thresholds = model_statistics.model_f2_statistics(path.MODEL_PATH, val)
    all_f2 = get_model_cv()

    for all in all_label:
        f2 = {}
        f2["avg"] = all[1]
        for i in range(13):
            f2[f"{i}"] = 0
        all_f2[get_model_identifier(all[0])] = f2

    for label in range(13):
        for one in one_label[label]:
            identifier = get_model_identifier(one[0])
            if one[1] > all_f2[identifier][f"{label}"]:
                all_f2[identifier][f"{label}"] = one[1]
                all_f2[identifier][f"model{label}"] = get_epoch_identifier(one[0])

    for key in all_f2.keys():
        avg = 0
        for i in range(13):
            avg += all_f2[key][f"{i}"]
        avg /= 13
        all_f2[key]["avg"] = avg

    save_model_cv(all_f2)


def build_model_test():
    epoch_test = get_epoch_test()
    model_cv = get_model_cv()
    model_test = {}
    for id in epoch_test.keys():
        model_test[id.split(".")[0]] = {"avg": 0}
        for i in range(13):
            model_test[id.split(".")[0]][f"{i}"] = 0

    for id in model_cv:
        value = model_cv[id]
        avg = 0
        for i in range(13):
            model = value[f"model{i}"]
            if epoch_test.get(model):
                test_model = epoch_test[model]
                test_f2 = test_model[f"{i}"]
                model_test[id.split(".")[0]][f"{i}"] = test_f2
                model_test[id.split(".")[0]][f"model{i}"] = model
                avg += test_f2
                model_test[id.split(".")[0]]["avg"] = avg / 13

    model_test_sorted = sorted(model_test.items(), key=lambda x: x[1]["avg"], reverse=True)
    model_test = {}
    for item in model_test_sorted:
        model_test[item[0]] = item[1]

    save_model_test(model_test)

    print("ok")


def build_threshold_cv():
    threshold_cv = {}

    for val in range(1, 6):
        _, _, thresholds = model_statistics.model_f2_statistics(path.MODEL_PATH, val)
        print("ok")

        for label in range(13):
            for weight_path in thresholds[label].keys():
                if not threshold_cv.get(get_epoch_identifier(weight_path)):
                    threshold_cv[get_epoch_identifier(weight_path)] = {}
                threshold_cv[get_epoch_identifier(weight_path)][f"{label}"] = thresholds[label][weight_path]
        print("ok")

    save_threshold_cv(threshold_cv)

def build_global_cv():
    epoch_cv = get_epoch_cv()
    global_cv = {"avg": 0}
    for i in range(13):
        global_cv[f"{i}"] = 0

    for id in epoch_cv.keys():
        model = epoch_cv[id]
        for label in range(13):
            if model[f"{label}"] > global_cv[f"{label}"]:
                global_cv[f"{label}"] = model[f"{label}"]
                global_cv[f"model{label}"] = id

    avg = 0
    for i in range(13):
        avg += global_cv[f"{i}"]
    global_cv["avg"] = avg / 13
    save_global_cv(global_cv)


def build_global_test():
    global_cv = get_global_cv()
    epoch_test = get_epoch_test()
    global_test = {"avg": 0}
    for i in range(13):
        global_test[f"{i}"] = 0

    for label in range(13):
        cv_model = global_cv[f"model{label}"]
        test_model = epoch_test[cv_model]
        f2 = test_model[f"{label}"]
        global_test[f"{label}"] = f2

    avg = 0
    for i in range(13):
        avg += global_test[f"{i}"]
        global_test["avg"] = avg / 13

    save_global_test(global_test)


def cnn_result_name_to_epoch_name(cnn: str):
    cnn = cnn[:-12]
    cnn = cnn.replace("-", "\\")
    return "\\" + cnn


def build_ensemble_epoch_cv():
    result_paths = []
    result_files = {}

    threshold_cv = get_threshold_cv()

    for root, dirs, files in os.walk(path.CNN_RESULT_PATH):
        for file in files:
            if "val2" not in file:
                continue
            if file.split(".")[-1] != "npy":
                continue
            result_paths.append(os.path.join(root, file))
            result_files[os.path.join(root, file)] = file

    for result_path in result_paths:
        file_name = result_files[result_path]
        epoch_name = cnn_result_name_to_epoch_name(file_name)
        y_true = get_test_labels()
        y_pred = np.load(result_path)

        threshold = threshold_cv[epoch_name]

        for i in range(13):
            y_pred[:, i] = y_pred[:, i] > threshold[f"{i}"]

        y_pred = y_pred.astype(np.int8)

        build_epoch_test(y_true, y_pred, None, identifier=epoch_name)


def build_threshold_test():
    result_paths = []
    result_files = {}

    threshold_test = {}

    for root, dirs, files in os.walk(path.CNN_RESULT_PATH):
        for file in files:
            if "val2" not in file:
                continue
            if file.split(".")[-1] != "npy":
                continue
            result_paths.append(os.path.join(root, file))
            result_files[os.path.join(root, file)] = file

    for result_path in result_paths:
        file_name = result_files[result_path]
        epoch_name = cnn_result_name_to_epoch_name(file_name)

        if threshold_test.get(epoch_name):
            continue
        print(f"epoch name {epoch_name}")
        y_true = get_test_labels()
        y_pred = np.load(result_path)
        threshold_test[epoch_name] = {}
        for i in range(13):
            score, threshold = metrics.greedy_f2_score(y_true[:, i], y_pred[:, i], 1, step=100)
            threshold_test[epoch_name][f"{i}"] = threshold[0]

        save_threshold_test(threshold_test)


def build_epoch_test_standard():
    result_paths = []
    result_files = {}

    threshold_test = get_threshold_test()
    epoch_test_standard = {}

    for root, dirs, files in os.walk(path.CNN_RESULT_PATH):
        for file in files:
            if "val2" not in file:
                continue
            if file.split(".")[-1] != "npy":
                continue
            result_paths.append(os.path.join(root, file))
            result_files[os.path.join(root, file)] = file

    for result_path in result_paths:
        file_name = result_files[result_path]
        epoch_name = cnn_result_name_to_epoch_name(file_name)
        threshold = threshold_test[epoch_name]
        y_true = get_test_labels()
        y_pred = np.load(result_path)

        for i in range(13):
            y_pred[:, i] = y_pred[:, i] > threshold[f"{i}"]

        y_pred = y_pred.astype(np.int8)

        one_label_greedy_f2_all = []
        for i in range(13):
            one_label_greedy_f2_all.append(fbeta_score(y_true[:, i], y_pred[:, i], beta=2))

        f2 = {"avg": np.mean(one_label_greedy_f2_all)}
        for i in range(13):
            f2[f"{i}"] = one_label_greedy_f2_all[i]

        epoch_test_standard[epoch_name] = f2
    save_epoch_test_standard(epoch_test_standard)


if __name__ == "__main__":
    print("ok")
    build_threshold_cv()
    # build_ensemble_epoch_cv()
    # build_threshold_test()
    # build_epoch_test_standard()
    # build_threshold_cv()
    # build_global_test()
    # build_global_cv()
    # build_model_test()
    # build_model_cv(2)
    # build_epoch_cv(2)
    # get_ablation_experiment_predict(path.MODEL_PATH, 2)
    # calc_xgb_f2_score()
    # get_existed_cnn_f2_score(1, path.MODEL_PATH)
