import torch
import matplotlib.pyplot as plt
from data_loader import get_image_path


def evaluate_image(model, dataset, index, device='cpu'):
    """
    评估单张图片
    
    Args:
        model: 训练好的模型
        dataset: 数据集对象
        index: 图片索引
        device: 计算设备
    
    Returns:
        predicted_label: 预测的标签
        true_label: 真实的标签
        confidence: 预测置信度
        file_path: 文件路径
        file_name: 文件名
    """
    # 检查索引是否有效
    if index < 0 or index >= len(dataset):
        print(f"错误：索引 {index} 超出范围。数据集大小为 {len(dataset)}")
        return None, None, None, None, None
    
    # 获取图片和标签
    image, true_label = dataset[index]
    
    # 添加批次维度并移动到设备
    image = image.unsqueeze(0).to(device)
    
    # 模型预测
    model.eval()
    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        predicted_label = predicted.item()
        confidence = confidence.item()
    
    # 获取文件路径信息
    file_path, file_name, images_path, labels_path = get_image_path(index, dataset)
    
    # 显示结果
    print("\n" + "=" * 50)
    print(f"图片索引: {index}")
    print(f"真实标签: {true_label.item()}")
    print(f"预测标签: {predicted_label}")
    print(f"预测置信度: {confidence * 100:.2f}%")
    print(f"文件名: {file_name}")
    print(f"数据文件路径: {images_path}")
    print(f"标签文件路径: {labels_path}")
    print("=" * 50)
    
    # 显示图片（需要反归一化）
    image_display = image.squeeze().cpu().numpy()
    # 反归一化：mean=0.1307, std=0.3081
    image_display = image_display * 0.3081 + 0.1307
    image_display = image_display.clip(0, 1)  # 确保值在 [0, 1] 范围内
    
    plt.figure(figsize=(6, 6))
    plt.imshow(image_display, cmap='gray')
    plt.title(f'索引: {index} | 真实: {true_label.item()} | 预测: {predicted_label} | 置信度: {confidence*100:.2f}%')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return predicted_label, true_label.item(), confidence, file_path, file_name

