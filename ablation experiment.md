# ablation experiment



## 绘图

+ model_cv_test_xxx
  + model_cv_test_info_diff：以模型为粒度，对比cv和test的差异
  + model_xception/resnet50/vgg19…… _cv_test_info_diff：在上述基础上，基于不同的模型，进行分别统计，查看模型的差异
  + model_cv_test_topn_info_diff：model_cv.json中，排名前N的模型，拿出来对比，去掉部分很差的模型带来的噪声，N可以取5， 10，15， 20，对应集成的数量。
+ epoch_cv_test_xxx
  + epoch_cv_test_info_diff：以epoch为粒度，对比cv、test的差异
  + epoch_xception/resnet50/vgg19…… _cv_test_info_diff：在上述基础上，基于不同的模型，进行分别统计，查看模型的差异
+ ensemble_cv_test_xgb_topn_diff: xgboost的topn，在cv和test上的差异，此处需要注意，xgboost有三种集成方式，应该分别对比，包括 cnn[avg], xgb[avg], xgb，分别是cnn取平均、xgb取平均、全投票。
+ ensemble_cv_test_cnn_topn_diff: cnn bagging的topn，在cv和test上的差异 
+ ensemble_cnn_xgb_topn_diff：取topn，对比cnn bagging和xgb stacking之间的区别。
+ xgb_singleval_kfold_topn_diff：对比使用单一val和k-fold之间的区别（多倍计算量的增益如何）
+ threshold_cv_test_info_diff：cv上的阈值和test上的阈值的差异
+ training_label_class_bar：训练集的数据分布
+ test_label_class_bar：测试集的训练分布

