import torch
from torchvision import datasets, transforms
import os


def get_data_loaders(batch_size=64):
    """
    加载 MNIST 数据集
    
    Args:
        batch_size: 批次大小
    
    Returns:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        test_dataset: 测试数据集（用于获取文件路径）
    """
    # 数据预处理：转换为张量并归一化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST 的均值和标准差
    ])
    
    # 下载并加载训练集
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    # 下载并加载测试集
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, test_loader, test_dataset


def get_image_path(index, dataset):
    """
    获取指定索引的图片路径信息
    
    Args:
        index: 图片索引
        dataset: 数据集对象
    
    Returns:
        file_path: 文件路径
        file_name: 文件名
    """
    # MNIST 数据集存储在 data/MNIST/raw/ 目录下
    data_dir = './data/MNIST/raw'
    
    # 根据是训练集还是测试集确定文件名前缀
    if dataset.train:
        prefix = 'train'
    else:
        prefix = 't10k'
    
    # MNIST 原始文件命名格式
    # 训练集: train-images-idx3-ubyte, train-labels-idx1-ubyte
    # 测试集: t10k-images-idx3-ubyte, t10k-labels-idx1-ubyte
    images_file = f'{prefix}-images-idx3-ubyte'
    labels_file = f'{prefix}-labels-idx1-ubyte'
    
    images_path = os.path.join(data_dir, images_file)
    labels_path = os.path.join(data_dir, labels_file)
    
    # 构建显示用的文件名
    file_name = f'{prefix}_image_{index}.png'
    file_path = os.path.join(data_dir, file_name)
    
    return file_path, file_name, images_path, labels_path

