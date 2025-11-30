import torch
import torch.nn as nn
import torch.optim as optim
from model import SimpleNet
from data_loader import get_data_loaders


def train_model(epochs, device='cpu'):
    """
    训练模型
    
    Args:
        epochs: 训练轮数
        device: 计算设备（'cpu' 或 'cuda'）
    
    Returns:
        model: 训练好的模型
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        test_dataset: 测试数据集
    """
    # 获取数据加载器
    train_loader, test_loader, test_dataset = get_data_loaders(batch_size=64)
    
    # 创建模型
    model = SimpleNet().to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"\n开始训练，共 {epochs} 个 epoch...")
    print("-" * 50)
    
    # 训练循环
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 统计
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        # 计算平均损失和准确率
        avg_loss = running_loss / len(train_loader)
        accuracy = 100 * correct / total
        
        print(f'Epoch [{epoch+1}/{epochs}] - '
              f'Loss: {avg_loss:.4f} - '
              f'Accuracy: {accuracy:.2f}%')
    
    print("-" * 50)
    print("训练完成！\n")
    
    # 在测试集上评估
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            test_total += target.size(0)
            test_correct += (predicted == target).sum().item()
    
    test_accuracy = 100 * test_correct / test_total
    print(f"测试集准确率: {test_accuracy:.2f}%\n")
    
    return model, train_loader, test_loader, test_dataset

