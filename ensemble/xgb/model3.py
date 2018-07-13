import os

from util import ensemble_util

model = ensemble_util.XGBoostModel(model_path=os.path.abspath(__file__),
                                   corr_threshold=0.9, search=20, top_n=10,
                                   eval_func=ensemble_util.xgb_greedy_f2_metric,
                                   xgb_param={
                                       'eta': [0.05, 0.1, 0.15, 0.2],
                                       'silent': True,  # option for logging
                                       'objective': 'binary:logistic',  # error evaluation for multiclass tasks
                                       'max_depth': range(3, 11)  # depth of the trees in the boosting process
                                   },
                                   number_round=1000,
                                   )

# model.train_all_label()
model.model_rank(10)