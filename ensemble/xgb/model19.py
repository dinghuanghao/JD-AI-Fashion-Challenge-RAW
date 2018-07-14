import os
import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
from sklearn.metrics import fbeta_score
from util import ensemble_util

model = ensemble_util.XGBoostModel(model_path=os.path.abspath(__file__),
                                   corr_threshold=0.9, search=5, top_n=2,
                                   xgb_param={
                                       'eta': [0.15],
                                       'silent': True,  # option for logging
                                       'objective': 'binary:logistic',  # error evaluation for multiclass tasks
                                       'max_depth': range(3, 8),  # depth of the trees in the boosting process
                                       'min_child_weight': [1]
                                   },
                                   number_round=1000,
                                   )

if __name__ == "__main__":

    model.train_all_label()

    data_x, data_y = model.build_all_datasets()
    print(data_x.shape)
    print(data_y[:4, :])
    pre_y = model.predict_real(data_x)
    print(pre_y[:4, :])

    print(data_y.shape)
    print(pre_y.shape)
    for i in range(13):
        print(fbeta_score(data_y[:, i], pre_y[:, i], beta=2))