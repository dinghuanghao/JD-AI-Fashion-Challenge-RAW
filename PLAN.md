# 待处理问题                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       

## 数据不均衡问题——紧急            

由于标签不均衡，第三个、倒数第五个标签的比例分别为 0.9，0.9，导致学习的过程中，这两个值基本被预测为1，其他的全部倾向于0，如下是使用Original、Segmented数据，训练了30个Epoch后，对Val进行预测的结果：

'my_output': array([[1.73255524e-14, 1.2 9659707e-03, 9 .99820888e-01, 1.13264077e-11,
        8.29777273e-05, 7.33385561e-04, 1.29288066e-10, 3.33989283e-06,
        9.99999642e-01, 1.45883618e-07, 7.95193245e-09, 6.16776527e-14,
        1.08028235e-12],
       [5.44229971e-24, 8.75639955e-07, 9.99993682e-01, 3.19490839e-18,
        3.62233954e-09, 5.08854610e-05, 2.00667231e-19, 7.88219073e-11,
        1.00000000e+00, 3.89001467e-13, 2.58106459e-19, 2.85204262e-21,
        1.46501005e-16],
       [5.82149884e-24, 3.34603919e-06, 9.99999762e-01, 2.74838875e-19,
        1.58705927e-06, 1.84531501e-07, 1.69889613e-17, 8.40869475e-12,
        1.00000000e+00, 2.02899695e-13, 8.78627700e-15, 4.77588761e-22,
        1.85213515e-18],
       [8.67193253e-22, 4.67927239e-05, 9.99993682e-01, 8.73316198e-19,
        6.74705298e-05, 3.18978823e-06, 2.79685578e-15, 4.86659379e-10,
        1.00000000e+00, 1.62092666e-11, 1.89955596e-16, 1.93475667e-21,
        2.95958373e-15],
       [1.09179207e-18, 8.63464447e-05, 9.99956965e-01, 1.27800927e-16,
        2.47708090e-06, 3.35021214e-05, 5.21968102e-13, 2.94599090e-07,
        1.00000000e+00, 1.27435701e-10, 1.18900603e-13, 3.04629966e-19,
        2.98748775e-15],
       [1.30868954e-14, 2.51155780e-05, 9.99815762e-01, 3.56874686e-12,
        5.91327589e-05, 1.60903975e-04, 4.33566946e-13, 4.47841515e-08,
        9.99980330e-01, 1.62867908e-09, 7.31504510e-11, 3.66607948e-13,
        2.99221433e-12],



可能的方案：

1. 权重，根据标签的密度来给每一种标签的loss加权重（使用加权bce作为权重）

2. 标签分开训练，将比例差不多的标签放到一起训练（无法学到标签之间的关联关系）

3. 虽然部分标签的值很小，但是仍然是有变化的，依然可用使用阈值来进行切割。可用试试切割是否准确



## Weighted BCE Loss

这是我自己想的一种方法，不知道是否有效，从效果来看的确让标签的值变得多样化。但是不清楚，这究竟是正确的还是随机的噪声，等后续将基于权重搜索的F2-Score实现后，看一看最终的效果

1. 根据标签的密度，按照一下权重来对BCE LOSS进行加权（将标签的密度放大到1）

```
527, 12.8, 1.1, 210, 2.8, 6.18, 279.32, 40.5, 1.11, 7.7, 14.79, 43.9, 156
```

```
global steps = 15115

'my_output': array([[2.87615620e-02, 3.60131587e-08, 1.00000000e+00, 9.34116933e-13,
        7.78415084e-01, 1.18359709e-02, 5.13187466e-08, 6.53684959e-02,
        1.00000000e+00, 4.39451336e-10, 2.75671277e-06, 5.23333088e-04,
        2.88969254e-14],
       [9.84157920e-01, 4.70034574e-05, 1.00000000e+00, 5.12703657e-10,
        6.76649809e-01, 2.43684705e-02, 4.05071041e-05, 6.81642354e-01,
        1.00000000e+00, 8.52663551e-10, 2.56278756e-04, 4.89057020e-05,
        5.48346160e-11],
       [3.22679698e-01, 8.49908702e-07, 1.00000000e+00, 5.53154961e-12,
        2.88722247e-01, 4.33327377e-01, 6.45374093e-05, 7.92339742e-01,
        1.00000000e+00, 1.82979916e-11, 2.55159563e-07, 6.24645281e-07,
        5.25569779e-12],
       [8.02957173e-03, 1.26484210e-07, 1.00000000e+00, 6.99199650e-13,
        9.92807746e-01, 1.51375111e-03, 2.38902931e-06, 7.49298215e-01,
        1.00000000e+00, 5.00108066e-10, 6.98621420e-07, 9.11940951e-05,
        9.13620037e-14],
       [4.98848230e-01, 1.14626251e-04, 1.00000000e+00, 3.84805380e-17,
        4.78389939e-06, 9.03646946e-01, 3.59346275e-10, 8.00661892e-02,
        1.00000000e+00, 1.76480235e-08, 3.01785888e-11, 5.12778480e-03,
        6.03238975e-20] 
```


```
global steps = 21985
{'my_output': array([[9.99936819e-01, 2.95100165e-07, 1.00000000e+00, 2.53286229e-11,
        8.42167377e-01, 8.11851919e-01, 1.81431301e-06, 4.42770422e-01,
        1.00000000e+00, 1.35282128e-12, 1.47628310e-09, 6.69168831e-10,
        7.68692218e-22],
       [2.93428987e-01, 3.93031724e-03, 1.00000000e+00, 3.17963600e-19,
        5.91983462e-06, 3.28291953e-01, 1.38492933e-05, 1.82572054e-03,
        1.00000000e+00, 2.86169336e-13, 7.25760632e-11, 3.57592329e-02,
        2.86158307e-25],
       [6.17216349e-01, 2.78287730e-03, 1.00000000e+00, 1.76822532e-12,
        3.25193349e-03, 3.50383285e-04, 5.68054691e-02, 4.42319037e-03,
        1.00000000e+00, 2.92008995e-09, 4.91057150e-08, 2.97096949e-02,
        6.63382720e-14],
       [2.83043346e-06, 9.23963395e-09, 1.00000000e+00, 2.96028035e-10,
        4.52617407e-01, 1.81097828e-04, 3.59331267e-07, 8.44809599e-03,
        1.00000000e+00, 1.14195431e-09, 1.18340167e-05, 1.55193251e-04,
        1.25051159e-15],
       [8.88331890e-01, 1.87989990e-05, 1.00000000e+00, 1.10013967e-13,
        3.47082675e-01, 1.70409694e-01, 2.99944281e-06, 1.59343719e-01,
        1.00000000e+00, 5.07361687e-11, 7.09992121e-10, 3.64392214e-02,
        1.88906778e-19],
       [1.48144827e-04, 2.73992032e-06, 1.00000000e+00, 8.39061893e-13,
        6.21999800e-01, 9.58482027e-01, 3.96175710e-06, 9.56250057e-02,
        1.00000000e+00, 6.08331181e-12, 2.45630669e-11, 9.76795200e-10,
        2.44567903e-20],
       [2.86166113e-09, 3.52901863e-09, 1.00000000e+00, 2.57086441e-08,
        8.20045531e-01, 9.97779071e-01, 1.14259535e-12, 4.96110506e-02,
        1.00000000e+00, 3.89012780e-12, 2.48500481e-02, 9.35924394e-11,
        4.58551036e-19],
```

从上诉的几次预测来看，网络的输出不再是单调的两个标签为1，其他为0. f2-score从0.6逐步上升到0.7（还未做阈值搜素，做了之后可能会有所不同）。



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

+ 将学习率降低为0.0001，暂时没有出现NaN问题了



## Tensorflow 训练速度

目前的平均训练速度：

+ 使用Segmented数据，1.5 Step/Second
+ 使用Original数据，0.6 Step/Second



实验：

+ 对预训练网络的部分参数采用固定化，以减少全局需要训练的参数数量，InceptionV3网络可用稳定在 2.1 Step/Second



## 优化算法问题

+ SGD：在IncceptionV3中使用SGD，SGD(lr=1e-4, momentum=0.9)，效果还不错，可以进一步增大学习率试试



## 模型精度问题

ResNet50：

BCE Loss：可逐步优化到0.04以下，但是F2-Score经常NAN，且F2-Score很低（Val 集）

F2-Score Loss：短时间内优化到0.2，然后趋于平缓了，F2-Score metric在0.7~0.8左右（Val 集）

BCE+F2=SCORE：



实验：

+ 将学习率降低到0.0001，并使用BCE+F2-Score，训练过程的F2-Score可达到0.84，但是预测时只有0.56



## 阈值搜索问题

在进行Validation的时候，不应该直接看F2-Score，而应该先动态算出阈值，然后再计算



## 迁移学习问题

### estimator API修改模型结构问题

一开始freeze住预训练模型后，进行训练。然后打开部分预训练模型，发现抛出以下异常。在stackoverflow上查看了一下，问题是因为SGD+Momentum算法有内部参数，这导致打开预训练层后，对应的层数找不到对应的SGD参数。

更深层次的原有，可能是因为Tensorflow的Checkpoint机制要求，参数的完全匹配。Checkpoint里的参数既不能比当前Graph中的参数多，也不能少（两种情况下都会出问题）。而相比之下，keras的model加载，因为只保存了网络的权重，因此不会受优化算法的变量所影响。

可能的解决办法：

+ 手动修改checkpoint，添加缺少的参数，并初始化为0
  + 这个试了下，能够把checkpoint里的数据通过tensorflow api读出来，但是没找到怎么修改
+ 删掉模型重新训练（这个神没问题的，但是没撒意义）
+ 一开始的时候打开所有的层次，保存一个初始化的checkpoint，里面包含了所有层的SGD参数，然后再freeze住预训练层进行训练
  + 打开预训练层后，SGD有一些参数的shape发生了变化，对应不上了
+ 不使用Keras + Estimator，而是直接使用Keras，自己对模型进行控制



=======================================================================================
Total params: 8,100,429
Trainable params: 1,392,973
Non-trainable params: 6,707,456
__________________________________________________________________________________________________
training steps per epoch is : 1373
training epoch is :10
1th load training dataset: 43927 images
labels tensor name is IteratorGetNext:1
(?, 13)
2018-05-20 15:41:59.109209: I T:\src\github\tensorflow\tensorflow\core\common_runtime\gpu\gpu_device.cc:1435] Adding visible gpu devices: 0
2018-05-20 15:41:59.109473: I T:\src\github\tensorflow\tensorflow\core\common_runtime\gpu\gpu_device.cc:923] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-05-20 15:41:59.109693: I T:\src\github\tensorflow\tensorflow\core\common_runtime\gpu\gpu_device.cc:929]      0 
2018-05-20 15:41:59.109830: I T:\src\github\tensorflow\tensorflow\core\common_runtime\gpu\gpu_device.cc:942] 0:   N 
2018-05-20 15:41:59.110029: I T:\src\github\tensorflow\tensorflow\core\common_runtime\gpu\gpu_device.cc:1053] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 8806 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:26:00.0, compute capability: 6.1)
2018-05-20 15:41:59.672386: W T:\src\github\tensorflow\tensorflow\core\framework\op_kernel.cc:1318] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key training/SGD/Variable_10 not found in checkpoint
Traceback (most recent call last):
  File "F:\Develop\Conda\lib\site-packages\tensorflow\python\client\session.py", line 1322, in _do_call
    return fn(*args)
  File "F:\Develop\Conda\lib\site-packages\tensorflow\python\client\session.py", line 1307, in _run_fn
    options, feed_dict, fetch_list, target_list, run_metadata)
  File "F:\Develop\Conda\lib\site-packages\tensorflow\python\client\session.py", line 1409, in _call_tf_sessionrun
    run_metadata)
tensorflow.python.framework.errors_impl.NotFoundError: Key training/SGD/Variable_10 not found in checkpoint
	 [[Node: save/RestoreV2 = RestoreV2[dtypes=[DT_FLOAT, DT_INT64, DT_FLOAT, DT_FLOAT, DT_FLOAT, ..., DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT], _device="/job:localhost/replica:0/task:0/device:CPU:0"](_arg_save/Const_0_0, save/RestoreV2/tensor_names, save/RestoreV2/shape_and_slices)]]
	 [[Node: save/RestoreV2/_101 = _Recv[client_terminated=false, recv_device="/job:localhost/replica:0/task:0/device:GPU:0", send_device="/job:localhost/replica:0/task:0/device:CPU:0", send_device_incarnation=1, tensor_name="edge_84_save/RestoreV2", tensor_type=DT_FLOAT, _device="/job:localhost/replica:0/task:0/device:GPU:0"]()]]

During handling of the above exception, another exception occurred:

………………

Process finished with exit code 1

### include top 和非 include top

除了最后两层外，前面的参数是否存在差异？似乎使用include top的训练效果更好（手动去掉后两层）