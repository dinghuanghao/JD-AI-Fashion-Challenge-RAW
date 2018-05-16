##准备阶段

+ DataLoader，异步、逐步地将图片加载到内存中

  + tf.api，它使用C++库编写的，比python的多线程以及队列等实现更为高效。但是tf.api和keras似乎还不能很好的集成。

  + TFRECORED格式已经和Tensorflow进行了集成，可用考虑

    可用后面通过TensorBoard看一看输入管道的使用情况，

  + 自己编写DataLoader

+ 检查图片的正确性

  + 发现下载的图片中有一两张在图像分割的时候报错了，需要排查一下

+ tensorflow board

  + 查看队列使用情况是否高效，即是否最大化了GPU的能力

+ 对比Tensorflow的高性能模型和Keras，看有多大的差别，如果差别不大则用keras，差别较大则用tensorflow



## 训练阶段
