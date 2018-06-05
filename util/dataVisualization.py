# -*- coding:utf-8 -*-
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pylab import mpl

from util import data_loader
from util import path
from util import data_metric

label_class_type_name = ["WearScene", "AreaStyle", "AgeRange", "Others"]
label_class_name = ["Sport", "Leisure", "OL/commuting", "JapaneseStyle", "KoreanStyle", "EuropeStyle", "EnglandStyle",
                    "Maid", "Debutante/Lady", "Simple", "Nature", "Street/Punk", "Nation"]

DATA_FILE = path.ORIGINAL_TRAIN_IMAGES_PATH

def get_columns():
    columns = []
    column_label_class_type_name = []
    for i in range(4):
        for j in range(data_metric.label_class_type_interval[i] - (data_metric.label_class_type_interval[i-1] if i!=0 else -1)):
            column_label_class_type_name.append(data_metric.label_class_type_name[i])
    columns.append(column_label_class_type_name)
    columns.append(data_metric.label_class_name)
    return columns

def get_frame():
    data = data_loader.get_labels(data_loader.list_image_name(DATA_FILE))
    data = np.array(data)
    df = pd.DataFrame(data, columns=get_columns())
    return df

def get_label_class_sum():
    df = get_frame()
    return df.sum(axis=0)

def show_label_class_bar():
    plt.figure(1,figsize=(10,6))
    sns.barplot(y=label_class_name, x=get_label_class_sum().tolist(), orient='h')
    plt.show()

if __name__ == "__main__":
    show_label_class_bar()

