# 命令行代码使用说明

    --源代码使用的是猫狗数据集
    --本代码可以更换不同的网络模型训练自己的数据集
    --可以控制训练轮数、批次大小等参数，支持断点续训练
    --可以输出不同格式的模型文件pt, onnx(静态/动态)
    --可以使用这些模型文件对图片或者图片集进行预测，并打印预测结果

## train 代码

```python
python main.py train
python main.py train --data_path ../model_train/data/dogs_cats/train --epochs 10 --batch_size 64 --model_name efficientnet_b0 --num_classes 10 --output_folder output --resume
# 加载自己的训练集； 训练轮数 ； 批次大小 ； 网络模型名称； 类别数 ； 输出文件夹 ；是否加载之前训练的模型 ；
```




## export 代码
```python
python main.py export
python main.py export --model_dir output/exp003/model/best.pth --format_type all
# 原模型的路径 ； 导出的格式 ；
```




## predict 代码
```python
python main.py predict
python main.py predict --model_path output/exp002/model/best.pt --img_path 路径到您的测试图片或图片文件夹
#  使用预测的模型的路径 ； 预测的图片/图片集的地址
```



## tensorboard可视化代码：

```python
tensorboard --logdir output/exp???/summary
```



## 支持的神经网络：

```python
ResNet系列（resnet18, resnet34, resnet50, resnet101, resnet152）
VGG系列（vgg16, vgg19）
DenseNet系列（densenet121, densenet169）
MobileNet系列（mobilenet_v2, mobilenet_v3_large, mobilenet_v3_small）
EfficientNet系列（efficientnet_b0, efficientnet_b7）
```



## 数据集格式说明：

```python
--train
----dogs
	----img1.jpg
    ----img2.jpg
    ...........
----cats
	----img1.jpg
    ----img2.jpg
    ...........
    
# 训练： 使用train文件夹
# 测试： 使用test文件夹
```

