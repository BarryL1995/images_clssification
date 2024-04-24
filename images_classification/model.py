import torch.nn as nn
from torchvision import models

"""
此函数现在支持更广泛的模型架构，能够根据模型名称动态地加载并调整预训练模型。它涵盖了大部分torchvision.models提供的主流模型，包括但不限于：

    ResNet系列（resnet18, resnet34, resnet50, resnet101, resnet152）
    VGG系列（vgg16, vgg19）
    DenseNet系列（densenet121, densenet169）
    MobileNet系列（mobilenet_v2, mobilenet_v3_large, mobilenet_v3_small）
    EfficientNet系列（efficientnet_b0, efficientnet_b7等）
"""


def initialize_advanced_model(model_name, num_classes):
    """
    初始化并调整指定的预训练模型架构，以适应给定数量的输出类别。

    参数:
    - model_name (str): 模型名称。
    - num_classes (int): 输出层的类别数。

    返回:
    - model (torch.nn.Module): 调整后的预训练模型。
    """
    # 使用getattr来动态获取模型构造函数和预训练权重
    model = None
    if model_name in models.__dict__:
        model_fn = getattr(models, model_name)
        if 'resnet' in model_name:
            model = model_fn(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif 'vgg' in model_name:
            model = model_fn(pretrained=True)
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        elif 'densenet' in model_name:
            model = model_fn(pretrained=True)
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        elif 'mobilenet_v2' in model_name:
            model = models.mobilenet_v2(pretrained=True)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        elif 'mobilenet_v3' in model_name:
            model = models.mobilenet_v3_large(pretrained=True)
            # MobileNetV3的分类器结构不同，它的全连接层是最后一个层
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
        elif 'efficientnet' in model_name:
            model = model_fn(pretrained=True)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        elif 'vit' in model_name:
            model = model_fn(pretrained=True)
            model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        else:
            raise ValueError(f"Unsupported or unrecognized model name: {model_name}")
    else:
        raise ValueError(f"Model name {model_name} is not present in torchvision.models")

    return model
