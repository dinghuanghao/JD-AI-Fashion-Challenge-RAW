from util import data_loader
from util import path

"""多标签分类数据集度量指标，可用于判断数据集划分是否正确"""

label_class_type_interval = [2, 6, 8, 12]  # 图片的大类别所对应的区间
label_class_type_name = ["穿着场景", "地区风格", "年龄段", "其他"]
label_class_name = ["运动", "休闲", "OL/通勤", "日系", "韩版", "欧美", "英伦",
                    "少女", "名媛/淑女", "简约", "自然", "街头/朋克", "民族"]


def label_set_num(label):
    """
    标签集合的数量，即训练样本的数量
    :param label: 
    :return: 
    """
    return label.shape[0]


def label_class_num(label):
    """
    标签的种类
    :param label: 
    :return: 
    """
    return label.shape[1]


def label_cardinality(label):
    """
    每个对象的平均标签
    :param label:
    :return:
    """
    return label[label == 1].size / label_set_num(label)


def label_density(label):
    """
    每个对象的平均标签密度
    :param label:
    :return:
    """

    return label_cardinality(label) / label_class_num(label)


def label_number_for_each_class(label):
    """
    每一种标签的数量
    :param label:
    :return:
    """
    lst = []
    for i in range(label_class_num(label)):
        a = label[:, i]
        lst.append(a[a == 1].size)

    return lst


def label_density_for_each_class(label):
    """
    每一种标签的密度
    :param label:
    :return:
    """
    lst = []
    for i in range(label_class_num(label)):
        a = label[:, i]
        lst.append(a[a == 1].size / label_set_num(label))

    return lst


def label_density_for_each_type(label):
    """
    四大类别的标签分别的密度
    :param label:
    :return:
    """
    den = label_density_for_each_class(label)
    a = 0
    interval_den = []
    for i in range(len(den)):
        a += den[i]
        if i in label_class_type_interval:
            interval_den.append(a)
            a = 0
    return interval_den


def label_diversity(label):
    """
    整个数据集中label的差异性（最大为2^n）
    :param label:
    :return:
    """

    dic = {}
    for i in label:
        dic["".join(str(e) for e in i)] = 1
    return len(dic)


def label_diversity_norm(label):
    """
    label差异性均值
    :param label:
    :return:
    """

    return label_diversity(label) / label_set_num(label)


def label_analysis(label):
    """
    标签分析，用于评估Class-Imbalance、数据集划分合理性
    :param label:
    :return:
    """
    print("对象平均标签数量: %f" % label_cardinality(label))
    print("对象平均标签密度: %f" % label_density(label))

    print("单个标签出现次数:")
    each_class = label_number_for_each_class(label)
    for i in range(len(label_class_name)):
        print("    %s: %f" % (label_class_name[i], each_class[i]))

    print("单个标签的平均出现次数:")
    each_class = label_density_for_each_class(label)
    for i in range(len(label_class_name)):
        print("    %s: %f" % (label_class_name[i], each_class[i]))

    print("单类标签的平均出现次数:")
    each_type = label_density_for_each_type(label)
    for i in range(len(label_class_type_name)):
        print("    %s: %f" % (label_class_type_name[i], each_type[i]))

    print("不同的标签组合数量: %f" % label_diversity(label))
    print("不同的标签组合密度: %f" % label_diversity_norm(label))


def labels_analysis(labels: []):
    print("对象平均标签数量:\t", "    ".join(str(i) for i in [label_cardinality(l) for l in labels]))
    print("================================================================")

    print("对象平均标签数量:\t", "    ".join(str(i) for i in [label_cardinality(l) for l in labels]))
    print("================================================================")

    print("单个标签的平均出现次数:")
    print("----------------------------------------------------------------")
    each_class = [label_density_for_each_class(l) for l in labels]
    for i in range(len(label_class_name)):
        print("    %s:\t" % label_class_name[i], "    ".join(str(j) for j in [value[i] for value in each_class]))
        print("----------------------------------------------------------------")

    print("================================================================")
    print("不同的标签组合数量:\t", "    ".join(str(i) for i in [label_diversity(l) for l in labels]))

    print("================================================================")
    print("不同的标签组合密度:\t", "    ".join(str(i) for i in [label_diversity_norm(l) for l in labels]))


if __name__ == '__main__':
    y, names = data_loader.load_label(path.ORIGINAL_TRAIN_IMAGES_PATH)
    # (name1, y1), (name2, y2), (name3, y3), (name4, y4), (name5, y5) = \
    #     data_loader.divide_data(names, y, (0.2, 0.2, 0.2, 0.2, 0.2))
    label_analysis(y)
