## 待处理

+  tensorflow训练加速
   +  从代码层面来分析有没有提速的可能性
+  模型loss函数优化
   +  当前使用BCE（Binary corss entropy），但比赛的评估方式是F2-Score，二者之间有一定差异
+  神经网络的浮点数输出转换为最终的标签
   +  阈值判定技术
   +  修改网络结构，使输出为1或者0
   +  当前输出神float32，导致可视化时的Accuracy不准确
+  图像分割数据集与原始数据集效果对比
   +  训练速度
   +  bias、variance
+  对F2-Score进行可视化
+  支持对Learning Rate、Learning Decay进行配置