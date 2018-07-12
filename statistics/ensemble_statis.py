import json
import os
import pathlib
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import config

from util import path

RECORD_DIR = os.path.join(os.path.abspath("."), "record", "ensemble")
pathlib.Path(RECORD_DIR).mkdir(parents=True, exist_ok=True)


def distplot(thresholds: [], subdir_name, save_file):
    plt.gcf().clear()
    ax = sns.distplot(thresholds)
    pathlib.Path(os.path.join(RECORD_DIR, subdir_name)).mkdir(parents=True, exist_ok=True)
    ax.get_figure().savefig(os.path.join(RECORD_DIR, subdir_name, save_file),  bbox_inches='tight')


def load_json(file) -> {}:
    with open(file, 'r') as f:
        return json.load(f)


def do_statistics(dir, model_names: []):
    f2_thresholds_all = []
    evaluate_jsons = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if "evaluate.json" not in file:
                continue
            skip = True
            for model_name in model_names:
                if model_name in root:
                    skip = False
            if skip:
                continue

            evaluate_json = load_json(os.path.join(root, file))
            f2_threshold = []
            max_depth = []
            eta = []
            min_child_weight = []
            for evaluate in evaluate_json.values():
                f2_threshold.append(evaluate["greedy_threshold"])
                f2_thresholds_all.append(evaluate["greedy_threshold"])
                eta.append(evaluate.get('eta', 0))
                min_child_weight.append(evaluate.get('min_child_weight', 0))
                max_depth.append(evaluate.get("max_depth", 0))
            distplot(f2_threshold, "threshold", "%s_.png" % os.path.split(root)[1])
            distplot(max_depth, "maxdepth", "%s.png" % os.path.split(root)[1])
            distplot(eta, "eta", "%s.png" % os.path.split(root)[1])
            distplot(min_child_weight, "min_child_weight", "%s.png" % os.path.split(root)[1])
    print(len(set(f2_thresholds_all)))


if __name__ == '__main__':
    do_statistics(path.XGBOOST_ENSEMBLE_PATH, ['model'])
