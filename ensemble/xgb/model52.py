import numpy as np

import os

from util import path
from util import ensemble_util

model = ensemble_util.XGBoostModel(model_path=os.path.abspath(__file__),
                                   corr_threshold=0.9, search=20, top_n=5,
                                   eval_func=ensemble_util.xgb_sparse_greedy_f2_metric,
                                   xgb_param={
                                       'eta': [0.1],
                                       'silent': True,  # option for logging
                                       'objective': 'binary:logistic',  # error evaluation for multiclass tasks
                                       'max_depth': range(2, 11),  # depth of the trees in the boosting process
                                       'min_child_weight': [1, 2, 3, 4, 5]
                                   },
                                   number_round=1000,
                                   )

# model.train_all_label()

test_x = model.build_test_datasets()

# output_avg表示是是否对xgboost同一个模型输出的多个数据进行平均
pre_y = model.predict_test(test_x, output_avg=True)
np.save(os.path.join(path.XGB_RESULT_PATH, "xgb_52_avg.npy"), pre_y)
model.save_submit(pre_y, "xgb_52_avg.txt")
