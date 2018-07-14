import os
import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
from sklearn.metrics import fbeta_score
from util import ensemble_util

model = ensemble_util.XGBoostModel(model_path=os.path.abspath(__file__),
                                   corr_threshold=0.9, search=5, top_n=2,
                                   xgb_param={
                                       'eta': [0.05],
                                       'silent': True,  # option for logging
                                       'objective': 'binary:logistic',  # error evaluation for multiclass tasks
                                       'max_depth': range(5, 8),  # depth of the trees in the boosting process
                                       'min_child_weight': [1]
                                   },
                                   number_round=1000,
                                   )

if __name__ == "__main__":
    model.model_merge(["model19", "model21"])
