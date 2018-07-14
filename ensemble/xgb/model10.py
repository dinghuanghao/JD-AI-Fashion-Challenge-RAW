import os

from util import ensemble_util

model = ensemble_util.XGBoostModel(model_path=os.path.abspath(__file__),
                                   corr_threshold=0.9, search=15, top_n=5,
                                   meta_model_dir="H:\\backup",
                                   xgb_param={
                                       'eta': [0.15],
                                       'silent': True,  # option for logging
                                       'objective': 'binary:logistic',  # error evaluation for multiclass tasks
                                       'max_depth': range(3, 8),  # depth of the trees in the boosting process
                                       'min_child_weight': [1]
                                   },
                                   number_round=1000,
                                   )

# model.train_all_label()
# model.model_rank(10)

model.get_meta_predict([3, 5], False)
