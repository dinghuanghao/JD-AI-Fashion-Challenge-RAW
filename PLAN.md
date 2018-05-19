# 待处理问题

## NAN问题

在evaluate和predict的时候，F2-Score为NaN，通过可视化发现，是模型的预测值，为NaN

+ 输出单元是Sigmoid，值域应该是 (0, 1)，说明在Sigmoid之前就出现了NAN
+ 网络采用预训练模型，仅仅修改了最后一层，因此应该不是网路结构异常
+ 该问题在使用BCE优化时容易出现，使用F2-SCORE优化时基本不出现
+ 只在针对没有见过的数据时会有这个问题？使用train数据进行evaluate试一下
+ 目前使用的是ResNet50，换一个模型试一下



查资料：

+ 大多数情况下是learning rate的问题
+ 可用通过tfdbg辅助查看问题
+ 通过summary权重来看是否有nan
+ 计算loss时对0、negative值进行下限剪切



实验：

+ 将Adam的学习率降低为0.0001，暂时没有出现NaN问题了

## Tensorflow 训练速度

目前的平均训练速度：

+ 使用Segmented数据，1.5 Step/Second

+ 使用Original数据，0.6 Step/Second

  

## 模型精度问题

ResNet50：

BCE Loss：可逐步优化到0.04以下，但是F2-Score经常异常，且F2-Score很低（Val 集）

F2-Score Loss：短时间内优化到0.2，然后趋于平缓了，F2-Score metric在0.7~0.8左右（Val 集）

BCE+F2=SCORE：



实验：

+ 将学习率降低到0.0001，并使用BCE+F2-Score，训练过程的F2-Score可达到0.84，但是预测时只有0.56