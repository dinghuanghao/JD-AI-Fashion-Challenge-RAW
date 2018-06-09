import re
import os
from util import path


def find_best_model(path, label_number=13, save_dir=None):
    evaluate_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if "evaluate" in file and "evaluate_revise" not in file:
                evaluate_files.append(os.path.join(root, file))

    all_label = {}
    one_label = {i: {} for i in range(label_number)}
    for file in evaluate_files:
        with open(file, "r") as f:
            with open(os.path.join(os.path.dirname(file), "evaluate_revise.txt"), "w+") as f_revise:
                weight_file = ""
                for i in f.readlines():
                    if "weight" in i:
                        weight_file = i.split(":")[-1].strip()
                        if os.path.dirname(file) == os.path.dirname(weight_file):
                            f_revise.write(i)
                        else:
                            f_revise.write(
                                "weight: %s\n" % os.path.join(os.path.dirname(file), os.path.basename(weight_file)))
                    else:
                        f_revise.write(i)

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
                        one_label[int(label)][weight_file] = float(greedy_f2)

        # 修复evaluate文件
        os.remove(file)
        os.rename(os.path.join(os.path.dirname(file), "evaluate_revise.txt"), file)

    all_label = sorted(all_label.items(), key=lambda x: x[1], reverse=True)
    for i in range(len(one_label)):
        one_label[i] = sorted(one_label[i].items(), key=lambda x: x[1], reverse=True)

    if save_dir is not None:
        with open(save_dir, "w") as f:
            f.write("all label\n")
            for i in all_label:
                f.write("%f: %s\n" % (i[1], i[0]))

            for i in range(len(one_label)):
                f.write("\n\n\n\n\none label: %d\n" % i)
                for j in one_label[i]:
                    f.write("%f: %s\n" % (j[1], j[0]))

    return all_label, one_label


find_best_model(path.MODEL_PATH, 13, "statistics.txt")
