import os
import re
import numpy as np
from sklearn.metrics import fbeta_score
from util import keras_util
from util import path
from util import model_statistics
from util import metrics

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

    return  np.array(labels, np.int8)


def get_meta_predict(mode_path):
    y_true = get_test_labels()

    original_test_file = []
    cnt = 0
    with open(path.TEST_DATA_TXT, 'r') as f:
        for i in f.readlines():
            image_name = i.split(",")[0] + ".jpg"
            original_test_file.append(os.path.join(path.ORIGINAL_TEST_IMAGES_PATH, image_name))

    weight_files = []
    _, _, thresholds = model_statistics.model_f2_statistics(path.MODEL_PATH, 1)

    for root, dirs, files in os.walk(mode_path):
        for file in files:
            if not file.split(".")[-1] == "hdf5":
                continue
            model_num = re.match(r".*model([0-9]*).*", root).group(1)
            if int(model_num) < 100:
                continue
            weight_files.append(os.path.join(root, file))

    for weight_file in weight_files:
        print(f"weight file {weight_file}")
        unique_path = re.match(r".*competition[\\/]*(.*)", weight_file).group(1)
        identifier = "-".join(unique_path.split("\\"))
        print(f"id {identifier}")
        cnn_result_path = os.path.join(weight_file + "(test)")
        print(f"result {cnn_result_path}")
        if not os.path.exists(keras_util.get_prediction_path(cnn_result_path)):
            attr_get_model, attr_model_config = keras_util.dynamic_model_import(weight_file)
            model = attr_get_model(output_dim=len(attr_model_config.label_position), weights=None)
            model.load_weights(weight_file)
            attr_model_config.val_files = []
            for data_type in attr_model_config.data_type:
                if data_type == path.DATA_TYPE_ORIGINAL:
                    attr_model_config.val_files.append(original_test_file)

            y_pred = keras_util.predict(model, attr_model_config, verbose=1)

            for i in range(13):
                y_pred[:, i] = y_pred[:, i] > thresholds[i][weight_file]

            y_pred = y_pred.astype(np.int8)

            keras_util.save_prediction_file(y_pred, cnn_result_path)
        else:
            y_pred = np.load(keras_util.get_prediction_path(cnn_result_path))

        with open(os.path.join(path.RESULT_PATH, "test.txt"), "a+") as f:
            one_label_greedy_f2_all = []


            for i in range(13):
                one_label_greedy_f2_all.append(fbeta_score(y_true[:, i], y_pred[:, i], beta=2))

            f.write("\n\n")
            f.write("Weight: %s\n" % weight_file)
            f.write("Greedy F2-Score is: %f\n" % np.mean(one_label_greedy_f2_all))
            for i in range(13):
                f.write("[label %d] greedy-f2=%4f\n" % (i, one_label_greedy_f2_all[i]))

        print(f"need predict {cnt} model")

def calc_xgb_f2_score():
    y_true = get_test_labels()
    y_pred = get_xgb_result()

    with open(os.path.join(path.RESULT_PATH, "test.txt"), "a+") as f:
        one_label_greedy_f2_all = []

        for i in range(13):
            one_label_greedy_f2_all.append(fbeta_score(y_true[:, i], y_pred[:, i], beta=2))

        f.write("\n\n")
        f.write("Greedy F2-Score is: %f\n" % fbeta_score(y_true, y_pred, average="samples", beta=2))
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



if __name__ == "__main__":
    # get_meta_predict(path.MODEL_PATH)
    # calc_xgb_f2_score()
    get_existed_cnn_f2_score(1, path.MODEL_PATH)