import os
import warnings
from pathlib import Path
import numpy as np

import onnxruntime
import torch
from PIL import Image

from model import initialize_advanced_model
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import ImageFolder

# 忽略警告
warnings.filterwarnings("ignore")


# 定义预处理转换
def pre_process():
    transform = transforms.Compose(
        [
            # 随机裁剪图像，裁剪出来的图像为224x224大小
            # 这有助于模型学习到图像的不同区域，增加模型对图像局部特征的感受力
            transforms.RandomResizedCrop(224),
            # 随机水平翻转图像
            # 这是一种非常常见的数据增强技术，可以增加数据的多样性，对于大多数图像分类任务来说都是有益的
            transforms.RandomHorizontalFlip(),
            # 随机旋转图像
            # 这一步通过随机旋转图像来增加数据的多样性，有助于提高模型对于旋转变化的鲁棒性
            transforms.RandomRotation(degrees=15),
            # 调整图像的颜色属性
            # 对亮度、对比度、饱和度和色调进行随机调整，以进一步增加数据的多样性
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            # 将PIL图像或`numpy.ndarray`转换为tensor
            # 这一步通常是必须的，因为PyTorch模型是使用tensor进行计算的
            transforms.ToTensor(),
            # 标准化图像
            # 使用ImageNet数据集的均值和标准差对图像进行标准化
            # 这一步有助于加快模型收敛，同时使模型对输入数据的小变化更加稳定
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform

# 定义数据数据加载器


def dataloader(data_path, batch_size=None):
    if batch_size is None:
        batch_size = 64

    transform = pre_process()
    # 创建数据集
    dataset = ImageFolder(root=data_path, transform=transform)

    # 划分数据集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size  # 使用总长度减去训练集的长度来确保精确分割
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # 创建加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 返回训练数据集和验证数据集
    return train_loader, val_loader


# 模型恢复后定义返回值
def load_model(path, net):
    print(f"模型恢复: {path}")
    model = torch.load(path, map_location=device)
    net.load_state_dict(state_dict=model["net"].state_dict(), strict=True)
    start_epoch = model["epoch"]
    train_batch = model["train_batch"]
    val_batch = model["val_batch"]
    best_acc = model["best_acc"]
    return start_epoch, train_batch, val_batch, best_acc


def create_folder(folder_name):
    # 当前时间作为文件夹名字
    folder_name = folder_name
    # 创建带有当前时间的文件夹路径
    root_dir = Path(f"output/01/{folder_name}")
    root_dir.mkdir(parents=True, exist_ok=True)
    # 创建summary 和model 文件夹
    os.makedirs(root_dir / "summary", exist_ok=True)
    os.makedirs(root_dir / "model", exist_ok=True)
    # 设置最后和最佳模型的路径，保存.pth格式.pth,二进制格式）
    return root_dir


# 模型训练


def training(model, dataloader, epochs, root_dir):
    # 定义保存文件的路径
    check_dir = root_dir / "model"
    best_path = check_dir / "best.pth"
    last_path = check_dir / "last.pth"

    #  参数初始化
    total_epochs = epochs
    summary_interval_batch = 1
    save_interval_epoch = 2
    start_epoch = 0
    train_batch = 0
    val_batch = 0
    best_acc = -1.0
    batch_train_loss = batch_val_loss = batch_train_acc = batch_val_acc = []
    # 数据加载
    train_loader, val_loader = dataloader

    # 模型加载
    net = model.to(device)
    # 定义交叉熵损失函数和自适应优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # 模型恢复， 继续训练
    if best_path.exists():
        start_epoch, train_batch, val_batch, best_acc = load_model(best_path, net)
    elif last_path.exists():
        start_epoch, train_batch, val_batch, best_acc = load_model(last_path, net)

    # 模型可视化输出
    writer = SummaryWriter(log_dir=str(root_dir / "summary"))
    writer.add_graph(net, torch.randn(1, 3, 224, 224).to(device))

    # 开始模型的训练
    for epoch in range(start_epoch, total_epochs + start_epoch):
        net.train()
        for train_batch, (imgs, labels) in enumerate(train_loader):
            # 将数据放到device上
            imgs = imgs.to(device)
            labels = labels.to(device)

            # 前向传播
            scores = net(imgs)
            train_loss = criterion(scores, labels)
            train_acc = torch.mean(
                (torch.argmax(scores, dim=1) == labels).to(torch.float)
            )

            # 反向传播
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            # 记录训练损失和准确率
            train_loss = train_loss.item()
            train_acc = train_acc.item()
            batch_train_loss.append(train_loss)
            batch_train_acc.append(train_acc)

            # 每间隔interval个batch打印并可视化一次训练信息
            if (train_batch + 1) % summary_interval_batch == 0:
                print(
                    f"epoch:{epoch}, train batch:{train_batch}, loss:{train_loss:.3f}, acc:{train_acc:.3f}"
                )
                writer.add_scalar("train_loss", train_loss, global_step=train_batch)
                writer.add_scalar("train_acc", train_acc, global_step=train_batch)

        net.eval()
        with torch.no_grad():
            for val_batch, (imgs, labels) in enumerate(val_loader):
                imgs = imgs.to(device)
                labels = labels.to(device)

                # 前向传播
                scores = net(imgs)
                val_loss = criterion(scores, labels)
                val_acc = torch.mean(
                    (torch.argmax(scores, dim=1) == labels).to(torch.float)
                )

                # 转化为标量并记录训练损失和准确率
                val_loss = val_loss.item()
                val_acc = val_acc.item()
                batch_val_loss.append(val_loss)
                batch_val_acc.append(val_acc)

                # 每间隔interval个batch打印并可视化一次验证信息
                if (val_batch + 1) % summary_interval_batch == 0:
                    print(
                        f"epoch:{epoch}, val batch:{val_batch}, loss:{val_loss:.3f}, acc:{val_acc:.3f}"
                    )
                    writer.add_scalar("val_loss", val_loss, global_step=val_batch)
                    writer.add_scalar("val_acc", val_acc, global_step=val_batch)

        # epoch模型的可视化
        epoch_train_loss = np.mean(batch_train_loss)
        epoch_train_acc = np.mean(batch_train_acc)
        epoch_val_loss = np.mean(batch_val_loss)
        epoch_val_acc = np.mean(batch_val_acc)
        print(50 * "=")
        print(
            f"epoch:{epoch}, epoch_train_loss:{epoch_train_loss:.3f}, epoch_train_acc:{epoch_train_acc:.3f}, val_loss:{epoch_val_loss:.3f}, val_acc:{epoch_val_acc:.3f}"
        )
        print(50 * "=")
        writer.add_scalars(
            "epoch_loss",
            {"train": epoch_train_loss, "val": epoch_val_loss},
            global_step=epoch,
        )
        writer.add_scalars(
            "epoch_acc",
            {"train": epoch_train_acc, "val": epoch_val_acc},
            global_step=epoch,
        )
        writer.close()

        # 模型持久化保存

        # 保存最优模型
        if val_acc > best_acc:
            best_acc = val_acc
            best_obj = {
                "net": net,
                "epoch": epoch,
                "train_batch": train_batch,
                "val_batch": val_batch,
                "best_acc": best_acc,
            }
            torch.save(best_obj, best_path.absolute())

        # 每隔一段保存一个模型
        if epoch % save_interval_epoch == 0:
            interval_obj = {
                "net": net,
                "epoch": epoch,
                "train_batch": train_batch,
                "val_batch": val_batch,
                "best_acc": best_acc,
            }
            torch.save(interval_obj, last_path.absolute())

        # 最终的模型保存
        last_obj = {
            "net": net,
            "epoch": epoch,
            "train_batch": train_batch,
            "val_batch": val_batch,
            "best_acc": best_acc,
        }
        torch.save(last_obj, last_path.absolute())


# 模型的格式转换
def export_model(format_type="all"):
    """
    NOTE: 可以通过netron（https://netron.app/）来查看网络结构
    将训练好的模型转换成可以支持多平台部署的结构，常用的结构：
    pt: Torch框架跨语言部署的结构
    onnx: 一种比较通用的深度学习模型框架结构
    tensorRT: 先转换成onnx，然后再进行转换使用TensorRT进行GPU加速
    openvino: 先转换为onnx，然后再进行转换使用OpenVINO进行GPU加速
    :param model_dir: model path
    :return: 不同格式的模型文件
    """

    # 加载保存的模型
    net = torch.load(model_dir / "best.pth", map_location=device)["net"]
    net.eval()
    example = torch.randn(1, 3, 224, 224).to(device)

    supported_formats = ["pt", "static_onnx", "dynamic_onnx"]

    if format_type not in supported_formats and format_type != "all":
        print(
            f"Unsupported format: {format_type}. Please choose from {supported_formats} or 'all'."
        )
        return

    formats_to_export = supported_formats if format_type == "all" else [format_type]

    for fmt in formats_to_export:
        if fmt == "pt":
            traced_script_module = torch.jit.trace(net, example)
            traced_script_module.save(os.path.join(model_dir, "best.pt"))
            print("Model exported in PT format.")

        elif fmt == "static_onnx":
            torch.onnx.export(
                model=net,
                args=example,
                f=os.path.join(model_dir, "model_static.onnx"),
                input_names=["images"],
                output_names=["scores"],
                opset_version=12,
                dynamic_axes=None,  # 指定非动态结构
            )
            print("Model exported in static ONNX format.")

        elif fmt == "dynamic_onnx":
            torch.onnx.export(
                model=net,
                args=example,
                f=os.path.join(model_dir, "model_dynamic.onnx"),
                input_names=["images"],
                output_names=["scores"],
                opset_version=12,
                dynamic_axes={"images": {0: "batch"}, "scores": {0: "batch"}},
            )
            print("Model exported in dynamic ONNX format.")


# 模型加载恢复
@torch.no_grad()
def predict(model_path, img_path, class_names):
    correct = 0
    total = 0

    # 自动判断模型格式
    model_format = "onnx" if model_path.endswith(".onnx") else "torch"

    # 加载模型
    if model_format == "torch":
        model = torch.load(model_path, map_location=device)
        if isinstance(model, dict):
            model = model["net"] if "net" in model else model
        model.eval().to(device)
    elif model_format == "onnx":
        model = onnxruntime.InferenceSession(model_path)
    else:
        raise ValueError("Unsupported model format!")

    # 加载图像或文件夹
    if os.path.isdir(img_path):
        dataset = ImageFolder(root=img_path, transform=pre_process())
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        class_names = dataset.classes
    else:
        img = Image.open(img_path).convert("RGB")
        img = pre_process()(img).unsqueeze(0).to(device)
        loader = [(img, torch.tensor(0))]  # Dummy label for single image
        if class_names is None:
            print("您需要测试的是单张图片, class_names is not defined")

    # 预测
    for i, (imgs, labels) in enumerate(loader):
        if model_format == "torch":
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
        elif model_format == "onnx":
            input_name = model.get_inputs()[0].name
            ort_inputs = {input_name: imgs.cpu().numpy()}
            ort_outs = model.run(None, ort_inputs)
            preds = np.argmax(ort_outs[0], axis=1)

        for idx, pred in enumerate(preds):
            predicted_label = class_names[pred] if model_format == "torch" else class_names[pred.item()]
            true_label = class_names[labels[idx]]
            is_correct = (true_label == predicted_label)
            print(f"Image {i+1}, True label: {true_label}, Predicted label: {predicted_label}, Correct: {is_correct}")
            correct += is_correct
            total += 1

    accuracy = correct / total
    print(f"\nAccuracy: {accuracy:.4f}")


if __name__ == "__main__":
    # 定义训练模型
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 加载数据集+
    dataloader = dataloader(r"images_classification\data\dogs_cats\train", batch_size=32)

    # 更改输出的类别数；输入是3通道RGB
    num_classes = 10  # 你的任务的类别数
    model_name = "efficientnet_b0"  # 选择的模型
    model = initialize_advanced_model(model_name, num_classes)

    """
    ResNet系列（例如resnet18, resnet34, resnet50, resnet101, resnet152）
    VGG系列（例如vgg16, vgg19）
    DenseNet系列（例如densenet121, densenet169）
    MobileNet系列（例如mobilenet_v2, mobilenet_v3_large, mobilenet_v3_small等）
    EfficientNet系列（例如efficientnet_b0, efficientnet_b7等）
    """
    # 创建保存模型的文件
    folder_name = "dog_cat_classification"
    root_dir = create_folder(folder_name)
    model_dir = root_dir / "model"
    # 调用不同功能的函数

    # 1. 训练
    training(model=model, dataloader=dataloader, epochs=10, root_dir=root_dir)

    # 2. 模型转换
    # export_model("all") # "pt",  "static_onnx",  "dynamic_onnx", "all"

    # 3. 加载模型测试数据
    # model_path = r"D:\pycharm_project\python_project0001\model\fc_model\images_classification\output\01\dog_cat_classification\model\best.pt"
    # img_path = r"C:\Users\83849\Desktop\test"
    # class_names = ["dog", "cat"]
    # predict(model_path, img_path, class_names)
