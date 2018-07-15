import os

from util import ensemble_util

model = ensemble_util.XGBoostModel(model_path=os.path.abspath(__file__),
                                   corr_threshold=0.9, search=20, top_n=5,
                                   meta_model_dir="E:\\backup\\jdfc",
                                   xgb_param={
                                       'eta': [0.2],
                                       'silent': True,  # option for logging
                                       'objective': 'binary:logistic',  # error evaluation for multiclass tasks
                                       'max_depth': range(4, 5),  # depth of the trees in the boosting process
                                       'min_child_weight': [1]
                                   },
                                   number_round=1000,
                                   )

# model.train_all_label()
# model.model_rank(10)
# model.get_meta_predict([1, 2, 3, 4, 5], False)
# model.find_segmented_model()

test_x = model.build_test_datasets()

# output_avg表示是是否对xgboost同一个模型输出的多个数据进行平均
pre_y = model.predict_test(test_x, output_avg=True)
model.save_submit(pre_y, model.file_name)

# data_x, data_y = model.build_all_datasets()
# print(data_x.shape)
# print(data_y[:4, :])
# pre_y = model.predict_real(data_x)
# print(pre_y[:4, :])
#
# print(data_y.shape)
# print(pre_y.shape)
# from sklearn.metrics import fbeta_score
# f2_marco = 0
# for i in range(13):
#     f2 = fbeta_score(data_y[:, i], pre_y[:, i], beta=2)
#     f2_marco += f2
#     print(f2)
# f2_marco /= 13
# print("average f2 is %6f" % f2_marco)