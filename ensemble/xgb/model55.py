import os
import numpy as np

from util import ensemble_util
from util import path

model = ensemble_util.XGBoostModel(model_path=os.path.abspath(__file__),
                                   corr_threshold=0.9, search=20, top_n=5,
                                   eval_func=ensemble_util.xgb_sparse_greedy_f2_metric,
                                   xgb_param={
                                       'eta': [0.2],
                                       'silent': True,  # option for logging
                                       'objective': 'binary:logistic',  # error evaluation for multiclass tasks
                                       'max_depth': range(2, 11),  # depth of the trees in the boosting process
                                       'min_child_weight': [1, 2, 3, 4, 5]
                                   },
                                   number_round=1000,
                                   )

# model.model_merge(["model51", "model52", "model53", "model54"])

model.build_and_predict_test()