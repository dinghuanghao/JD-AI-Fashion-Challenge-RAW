# README

epoch_cv:：在cross validation上，评估每个epoch的f2 score

epoch_test：使用cross validation上得出的阈值，在test集上评估每个epoch的f2 score

epoch_test_standard: 使用test集上搜索出来的阈值，在test集上求每个epoch的值。相当于是最优解。

global_cv：从cross validation中选出每个标签最优秀的模型，组装出一个最优的模型，基于cv集进行评估

global_test：使用cross validation中每个标签最优秀的模型，组装出最优的模型，基于test集进行评估

model_cv：在cross validation上，在每个模型的多个epoch中，选出最优秀的标签组装在一起

model_test：使用model_cv中的组装方式，在test集上进行评估

threshold_cv：cross validation上的阈值

threshold_test：test集上搜索出来的阈值

ensemble_cv：集成模型在cv集上的结果

ensemble_test：集成模型在test集上的结果
+ avg\[cnn\]：对cnn输出进行平均得单个样本，然后再输入xgboost得到五个样本，然后二值化并投票
+ avg\[xgb\]：对xgb的25个输出进行平均，得到五个输出，然后二值化并投票
+ 无后缀：对xgb的25个输出进行二值化并投票
