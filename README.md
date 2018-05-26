# README

## 使用指南

+ 运行path.py，生成一些必备的文件夹

+ 运行downloader.py下载图片到data/original/train目录下

+ （可选）运行segmentation.py，将下载的图片进行人像分割

+ 编写模型代码

  + 参照下方的 “模型”

+ 运行model文件，例如运行 model/estimator/resnet1/resnet50.py

+ 运行tensorboard查看训练状况

  通过cmd进入到model下的record目录，然后执行如下命令，并在浏览器中输入打印的URL

  ```bash
  tensorboard --logdir=.
  ```
  

## 模型

### 目录结构

+ model下分为estimator和keras，分别代表两种套API相关的模型

+ estimator下每个文件代表一个模型，模型下的record目录用于存储训练结果

+ keras下每个目录代表一类模型，然后问一个文件夹中包括了该类模型的不同实例（不同的超参数、不同的局部结构、不同的优化方式、不同的数据类型）。每个实例均有一个record目录用于保存所有结果。

  |------baseline

  ​	|-----1

  ​		|------record

  ​		|------baseline.py

  |------resnet50

  ​	|-----1

  ​		|------record

  ​		|------resnet50.py

  ​	|---------2

  ​		|------record

  ​		|------resnet50.py

  ​	|---------3

  ​		|------record

  ​		|------resnet50.py



keras下的目录结构为每一套超参数、细微的改动都创建一个新的文件夹，确保了代码和训练结果的一一对应。

### 模型编写

#### 模型编写

如果使用keras + estimator API那么可参考model/estimator/resnet50/resnet1/resnet50.py

+ 编写keras模型，并加载预训练模型
+ keras转为estimator，并配置checkpoint（模型自动保存）、summary（模型可视化间隔）
+ 修改模型的配置ModelConfig()，包含训练数据集、k-fold划分、epoch等，ModelConfig会自动保存到recored/n/目录下，记录每次训练的详细参数和起止时间。这样在不停修改参数并增量地训练时，可用对照Tensorboard曲线和时间段来分析参数的效果。



如果使用keras API那么可参考model/keras/baseline/1/baseline.py