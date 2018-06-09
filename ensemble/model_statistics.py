import re
import os
from util import path


def model_statistics(path, save_file=None):
    """
    对model目录下的所有包含"evaluate"字段的文件进行统计，分别得到all-label、one-label统计
    :param path: 需要统计的目录
    :param save_file: 输入文件
    :return:
    """
    evaluate_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if "evaluate" in file and "evaluate_revise" not in file:
                evaluate_files.append(os.path.join(root, file))

    all_label = {}
    one_label = {}
    for file in evaluate_files:
        with open(file, "r") as f:
            weight_file = ""
            for i in f.readlines():
                if "weight" in i:
                    weight_file = i.split(":")[-1].strip()

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


model_statistics(path.MODEL_PATH, "statistics.txt")
