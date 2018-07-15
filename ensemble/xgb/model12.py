import os

from util import ensemble_util

model = ensemble_util.XGBoostModel(model_path=os.path.abspath(__file__),
                                   corr_threshold=0.9, search=15, top_n=5,
                                   eval_func=ensemble_util.xgb_sparse_greedy_f2_metric,
                                   meta_model_dir="H:\\backup",
                                   xgb_param={
                                       'eta': [0.13, 0.15, 0.17],
                                       'silent': True,  # option for logging
                                       'objective': 'binary:logistic',  # error evaluation for multiclass tasks
                                       'max_depth': range(2, 8),  # depth of the trees in the boosting process
                                       'min_child_weight': [1, 2, 3]
                                   },
                                   number_round=1000,
                                   )

model.train_all_label()

data_x, data_y = model.build_all_datasets()
model.predict_real_f2(data_x, data_y)