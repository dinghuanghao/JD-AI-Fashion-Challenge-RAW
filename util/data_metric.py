from util import downloader
from util import image_loader

"""多标签分类数据集度量指标，可用于判断数据集划分是否正确"""


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


if __name__ == '__main__':
    x, y = image_loader.load_image(downloader.DEMO_TRAINING_PHOTOS_PATH)
    print(label_cardinality(y))
    print(label_density(y))
    print(label_diversity(y))
    print(label_diversity_norm(y))
