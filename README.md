# README

## 使用指南

+ 运行path.py，生成一些必备的文件夹

+ 运行downloader.py下载图片到data/original/train目录下

+ （可选）运行segmentation.py，将下载的图片进行人像分割

+ 编写模型代码

  + 参照下方的 “模型”

+ 运行main.py，进行训练

  + 修改main.py中的第六行、第七行，从不同的model获取Estimator以及Config
  + 执行main.py

+ 运行tensorboard查看训练状况

  通过cmd进入到目标目录，然后执行如下命令，并在浏览器中输入打印的URL

  ```bash
  tensorboard --logdir=.
  ```



## 模型

### 目录结构

model目录下的每一个子目录，表示一个模型，这样划分的目的是每一个模型都有自己的训练记录（不同输入大小、loss函数等，均算作不同的模型，因为他们之间无法直接共享参数，且如果混在一起之后，打破了参数调整的隔离性），便于独立分析每一种算法、每一套超参数，例如：

+ resnet1：采用resnet50模型，输入大小为224*224，去掉了最后的softmax层，新增一个sigmoid
  + record：训练记录，用于可视化和权重恢复
    + 1：第一份记录（创建ModelConfig时，通过record_sub_dir参数设置）
    + 2：第二份记录
    + ……
  + resnet50.py：模型的代码
+ resnet2：采用resnet50模型，输入大小为48*48，去掉了最后的softmax层，新增一个sigmoid
  + record：训练记录，用于可视化和权重恢复    
  + resnet50.py：模型的代码
+ ……

### 模型编写

#### 框架选型

经过tflayter、keras的对比后，最终选用keras + Estimator的方案。

Estimator是tensorflow官方推荐的一种模式，它有一些好处：

+ 会自动的对训练过程进行记录和自动加载
+ 支持分布式训练
+ ……

#### 模型编写

可参考model/resnet50/resnet1/resnet50.py

+ 编写keras模型，并加载预训练模型
+ keras转为estimator，并配置checkpoint（模型自动保存）、summary（模型可视化间隔）
+ 修改模型的配置ModelConfig()，包含训练数据集、k-fold划分、epoch等，ModelConfig会自动保存到recored/n/目录下，记录每次训练的详细参数和起止时间。这样在不停修改参数并增量地训练时，可用对照Tensorboard曲线和时间段来分析参数的效果。