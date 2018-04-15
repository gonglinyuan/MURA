## 注意事项

1. 最后要直接交inference的程序, 所以要保留好预处理数据的代码 (从原始数据到模型input全过程的code都要保留)
2. train set用来训练模型; valid set划分为valid1和valid2, 其中valid1用来对模型调参, valid2用来ensemble
3. 参数尽量分离到另外一个文件里 (目前是hyperparameters.py) , 调参的时候不要动代码

## Dataset

1. 每个study有若干个不同视角的图片, 判断整个study是positive还是negative

2. 已划分train和valid (大概10:1)

3. Positive : negative = 5810:8963 (train + valid)

   Positive : negative = 544:664 (valid)

4. 数据: MURA_trainval.zip CRC32: 3ef6712c

## 计算资源

1. 图灵班GPU服务器

   Titan Xp * 5 * 2 (与别人共享)

   Tensorflow1.4 + python3.6

   龚林源: http://162.105.162.67:40691

   马知遥: http://162.105.162.39:15108
   
   运行前先设置device, 不要一下子占了所有GPU

1. 美团云

   Tesla P40 * 4

   需要使用tf.gfile读写文件, 比较麻烦; 上传下载文件容易中断; 文件读写速度非常慢

   支持tensorflow1.4以下的版本, 支持python3.6以下的版本

   Keras配置后可以import, 不知道能不能用GPU跑起来

1. 吉如一哥哥的台式机

   GTX1080Ti, 只能短期借用

## 统一开发环境

1. 程序都能在python3.5+运行
2. 模型尽量能在Tensorflow1.4上面跑
   - Tensorflow的新版本可以导入旧版本的模型(前提是版本号第一位一样, 都是1)
   - Tensorflow旧版本和新版本的源程序可能不兼容(尽量保证兼容Tensorflow1.4)

## 初步结果

1. DenseNet121
   - Dropout (Dense Layers): drop_prob = 0.2
   - Valid set loss = 0.5463
   - Valid set acc = 0.8175
2. DenseNet121
   - Dropout (Output Layer): drop_prob = 0.5
   - Valid set loss = 1.4147
   - Valid set acc = 0.8081
3. DenseNet121

   - Dropout (Dense Layers): drop_prob = 0.1
   - Dropout (Output Layer): drop_prob = 0.5
   - Weight decay: 1e-3
   - Valid set loss = 0.6707
   - Valid set acc = 0.7755
4. DenseNet121
   - Freeze the initial convolution layer
   - Valid set loss = 1.056
   - Valid set acc = 0.8015

## Done

1. 按照论文1712.06957.pdf先做个Baseline

   DenseNet-121 (pretrained on ImageNet) 修改输出层

   算法可以参考1608.06993.pdf

   代码可以参考https://github.com/flyyufelix/cnn_finetune

## To-do

1. 对已有模型的调参 (freeze一些layer, 加regularization), 数据增强 (旋转, 剪切)

2. 写inference的代码 (TenCrop)

3. 尝试其它网络结构

   参考https://github.com/flyyufelix/cnn_finetune

4. 针对数据做一些处理 / 清洗
   - 去黑边
   - 避免不成比例的伸缩