import copy
import os

import xgboost as xgb

from util import ensemble_util


class XGBoostModel(ensemble_util.EnsembleModel):
    def __init__(self, xgb_param=None, number_round=None,
                 *args, **kwargs):
        super(XGBoostModel, self).__init__(*args, **kwargs)
        self.xgb_param = xgb_param
        self.number_round = number_round

    def save_model(self, model, model_name):
        model.save_model(os.path.join(self.record_dir, model_name))

    def get_model_name(self, val_index, label):
        return "ensemble_val%d_label%d.xgb" % (val_index, label)

    def train_all_label(self):
        for val_index in range(1, 6):
            for label in range(13):
                self.train_single_label(val_index=val_index, label=label)

    def train_single_label(self, val_index, label):
        train_x, train_y, val_x, val_y = self.build_datasets(val_index=val_index, target_label=label)
        data_train = xgb.DMatrix(data=train_x, label=train_y)
        data_val = xgb.DMatrix(data=val_x, label=val_y)

        evals = [(data_train, 'train'), (data_val, 'eval')]
        best_f2 = 0
        best_eta = 0
        best_max_depth = 0
        best_model = None
        best_pred = None
        best_xgb_param = None

        for eta in self.xgb_param["eta"]:
            for max_depth in range(self.xgb_param['max_depth'][0], self.xgb_param['max_depth'][1]):
                xgb_param = {
                    'eta': eta,
                    'silent': self.xgb_param['silent'],  # option for logging
                    'objective': self.xgb_param['objective'],  # error evaluation for multiclass tasks
                    'max_depth': max_depth,  # depth of the trees in the boosting process
                    'nthread': self.xgb_param['nthread'],
                }

                bst = xgb.train(xgb_param, data_train, self.number_round, evals=evals,
                                feval=ensemble_util.xgb_greedy_f2_metric,
                                early_stopping_rounds=10)

                data_eva = xgb.DMatrix(val_x)
                ypred = bst.predict(data_eva, ntree_limit=bst.best_ntree_limit)
                ypred = ypred.reshape((-1, 1))
                f2 = self.evaluate(y_pred=ypred, y=val_y, weight_name=self.get_model_name(val_index, label))
                self.save_log("eta:%f, max_depth:%d, f2:%f" % (eta, max_depth, f2))
                self.save_log("best_iteration:%4f,  best_score:%4f, best_ntree_limit=%4f" % (bst.best_iteration,
                                                                                             bst.best_score,
                                                                                             bst.best_ntree_limit))
                self.save_log("\n")
                if f2 > best_f2:
                    best_f2 = f2
                    best_model = bst
                    best_eta = eta
                    best_max_depth = max_depth
                    best_pred = ypred
                    best_xgb_param = copy.deepcopy(xgb_param)

        self.evaluate(y_pred=best_pred, y=val_y, weight_name=self.get_model_name(val_index, label),
                      xgb_param=best_xgb_param, save_evaluate=True)

        self.save_log("save best model for val[%d] label[%d], f2[%f] eta[%f] max_depth[%d]" %
                      (val_index, label, best_f2, best_eta, best_max_depth))

        self.save_model(best_model, self.get_model_name(val_index, label))


model = XGBoostModel(model_path=os.path.abspath(__file__),
                     corr_threshold=0.9, search=20, top_n=5,
                     xgb_param={
                         'eta': [0.05, 0.1, 0.15, 0.2],
                         'silent': True,  # option for logging
                         'objective': 'binary:logistic',  # error evaluation for multiclass tasks
                         'max_depth': [3, 10],  # depth of the trees in the boosting process
                         'nthread': 16,
                     },
                     number_round=1000,
                     )

# model.train_single_label(val_index=1, label=0)
model.train_all_label()
