import argparse
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

# 在这里导入或定义 training, export_model, predict 函数

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


def create_folder(base_dir="output"):
    # 检查基目录下已存在的文件夹，找到当前最大编号
    base_path = Path(base_dir)
    exp_folders = [p for p in base_path.iterdir() if p.is_dir() and p.name.startswith("exp")]
    max_num = 0
    for folder in exp_folders:
        try:
            num = int(folder.name.replace("exp", ""))
            if num > max_num:
                max_num = num
        except ValueError:
            continue  # 如果文件夹名不符合"expXXX"格式，忽略此文件夹

    # 创建新的文件夹，编号为当前最大编号+1
    new_folder_num = max_num + 1
    new_folder_name = f"exp{new_folder_num:03d}"  # 保持编号为三位数，如002, 011等
    new_folder_path = base_path / new_folder_name
    new_folder_path.mkdir(parents=True, exist_ok=True)

    # 在新文件夹内创建summary和model子文件夹
    os.makedirs(new_folder_path / "summary", exist_ok=True)
    os.makedirs(new_folder_path / "model", exist_ok=True)

    return new_folder_path


# 模型训练


def training(model, dataloader, epochs, root_dir, resume):

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

    if resume:
        # 模型恢复， 继续训练
        if best_path.exists():
            print(f"尝试从{best_path}恢复模型...")
            start_epoch, train_batch, val_batch, best_acc = load_model(best_path, net)
            print(f"模型恢复成功：从 epoch： {start_epoch} ，train_batch {train_batch}  继续训练...")
        elif last_path.exists():
            print(f"尝试从{best_path}恢复模型...")
            start_epoch, train_batch, val_batch, best_acc = load_model(last_path, net)
            print(f"模型恢复成功：从 epoch： {start_epoch} ，train_batch {train_batch}  继续训练...")
        else:
            print("未找到模型检查点，从头开始训练...")

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
# 模型的格式转换
def export_model(model_path, format_type="all"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 确保路径正确
    net = torch.load(model_path, map_location=device)["net"]
    net.eval()
    example = torch.randn(1, 3, 224, 224).to(device)

    # 获取目录和文件名
    model_dir = Path(model_path).parent
    model_filename = Path(model_path).name

    supported_formats = ["pt", "static_onnx", "dynamic_onnx"]

    if format_type not in supported_formats and format_type != "all":
        print(
            f"Unsupported format: {format_type}. Please choose from {supported_formats} or 'all'."
        )
        return

    formats_to_export = supported_formats if format_type == "all" else [format_type]

    for fmt in formats_to_export:
        output_path = model_dir / f"{model_filename}_{fmt}"
        if fmt == "pt":
            traced_script_module = torch.jit.trace(net, example)
            traced_script_module.save(output_path.with_suffix('.pt'))
            print("Model exported in PT format.")

        elif fmt in ["static_onnx", "dynamic_onnx"]:
            dynamic_axes = None if fmt == "static_onnx" else {"images": {0: "batch"}, "scores": {0: "batch"}}
            torch.onnx.export(
                model=net,
                args=example,
                f=output_path.with_suffix('.onnx'),
                input_names=["images"],
                output_names=["scores"],
                opset_version=12,
                dynamic_axes=dynamic_axes,
            )
            print(f"Model exported in {fmt} ONNX format.")


# 模型加载恢复
@torch.no_grad()
def predict(model_path, img_path, class_names):
    correct = 0
    total = 0

    # 自动判断模型格式
    model_path = str(model_path)
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
            # 确保预测索引在 class_names 范围内
            pred_index = pred.item() if isinstance(pred, torch.Tensor) else pred
            pred_index = min(max(pred_index, 0), len(class_names) - 1)
            predicted_label = class_names[pred_index]

            true_label = class_names[labels[idx]]
            is_correct = (true_label == predicted_label)
            print(f"Image {i + 1}, True label: {true_label}, Predicted label: {predicted_label}, Correct: {is_correct}")
            correct += is_correct
            total += 1
    accuracy = correct / total
    print(f"\nAccuracy: {accuracy:.4f}")


def get_args():
    parser = argparse.ArgumentParser(description="Image Classification Training, Exporting and Prediction")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # 训练子命令
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--data_path', type=str, default=r"../images_classification/data/dogs_cats/train", help='Path to the dataset')
    train_parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    train_parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    train_parser.add_argument('--model_name', type=str, default='efficientnet_b0', help='Model architecture')
    train_parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')
    train_parser.add_argument('--output_folder', type=str, default='output', help='Output folder for saving models')
    train_parser.add_argument('--resume', action='store_true', help='Resume training from the last checkpoint')
    # 导出模型子命令
    export_parser = subparsers.add_parser('export', help='Export the model')
    export_parser.add_argument('--model_dir', type=str, help='Directory of the trained model')
    export_parser.add_argument('--format_type', type=str, default='all', choices=['pt', 'static_onnx', 'dynamic_onnx', 'all'], help='Format to export the model')
    export_parser.add_argument('--output_folder', type=str, default='output', help='Output folder for models')
    # 预测子命令
    predict_parser = subparsers.add_parser('predict', help='Predict using the model')
    predict_parser.add_argument('--model_path', type=str, help='Path to the model for prediction')  # 去掉了默认值
    predict_parser.add_argument('--img_path', type=str, default=r"../images_classification/data/dogs_cats/test", help='Path to the image or directory for prediction')
    predict_parser.add_argument('--class_names', nargs='+', default=["dog", "cat"], help='List of class names for prediction')
    predict_parser.add_argument('--output_folder', type=str, default='output', help='Output folder for models')  # 可选，用于自动查找模型
    return parser.parse_args()


def find_best_or_latest_checkpoint(base_dir="output"):
    base_path = Path(base_dir)
    exp_folders = list(base_path.glob("exp*"))
    if not exp_folders:
        return None, None
    latest_exp_folder = max(exp_folders, key=lambda x: int(x.name.replace("exp", "")))
    best_model_path = latest_exp_folder / "model" / "best.pth"
    last_model_path = latest_exp_folder / "model" / "last.pth"
    if best_model_path.is_file():
        return latest_exp_folder, best_model_path
    elif last_model_path.is_file():
        return latest_exp_folder, last_model_path
    return latest_exp_folder, None


def main():
    global device  # 声明 device 为全局变量，以便在其他函数中使用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = get_args()

    if args.command == 'train':
        model = initialize_advanced_model(args.model_name, args.num_classes)
        train_loader, val_loader = dataloader(args.data_path, args.batch_size)

        if args.resume:
            latest_exp_folder, checkpoint_path = find_best_or_latest_checkpoint(args.output_folder)
            if checkpoint_path:
                print(f"恢复训练：使用检查点 {checkpoint_path}")
                root_dir = latest_exp_folder  # 使用包含最佳或最新检查点的文件夹
                training(model, (train_loader, val_loader), args.epochs, root_dir, args.resume)
            else:
                print("未找到有效的模型检查点，将从头开始训练...")
                root_dir = create_folder(args.output_folder)
                training(model, (train_loader, val_loader), args.epochs, root_dir, False)
        else:
            root_dir = create_folder(args.output_folder)
            print(f"开始新的训练实验，输出目录：{root_dir}")
            training(model, (train_loader, val_loader), args.epochs, root_dir, False)

    elif args.command == 'export':
        output_folder = getattr(args, 'output_folder', 'output')  # 使用默认值'output'，如果没有提供
        if args.model_dir:
            model_path = Path(args.model_dir)
            print(f"指定模型路径：{model_path}")
        else:
            _, model_path = find_best_or_latest_checkpoint(output_folder)
            if model_path is None:
                print("未找到最新的模型文件best.pth。")
                return
            print(f"未指定模型路径，自动选择最后保存的模型路径：{model_path}")
        export_model(model_path, args.format_type)

    elif args.command == 'predict':
        if args.model_path:
            model_path = Path(args.model_path)
            print(f"使用指定模型进行预测：{model_path}")
        else:
            # 如果没有指定模型路径，自动寻找最新的best.pth
            output_folder = getattr(args, 'output_folder', 'output')  # 如果未提供，使用默认值'output'
            _, model_path = find_best_or_latest_checkpoint(output_folder)
            if model_path is None or not model_path.is_file():
                print("未找到最新的模型文件best.pth，无法进行预测。")
                return
            print(f"自动选择模型进行预测：{model_path}")
        predict(model_path, args.img_path, args.class_names)


if __name__ == "__main__":
    main()
