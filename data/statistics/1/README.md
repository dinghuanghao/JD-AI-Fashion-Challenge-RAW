# statistics
存放图片的统计信息，用于在训练的时候对图像进行标准化。本目录下的所有统计，均基于1.txt(k_fold文件)进行数据划分

val_1_train_1_mean_224_224_['segmented'].npy
+ val index 为 1
+ 图片值scale比例为1（不对值进行归一化）
+ 求图片均值（mean）
+ 图片尺寸为224, 224
+ 仅仅对segmented数据进行处理

val_1_train_1_std_224_224_['segmented'].npy
+ val index 为 1
+ 图片值scale比例为1（不对值进行归一化）
+ 求图片的标准差（std）
+ 图片尺寸为224, 224
+ 仅仅对segmented数据进行处理