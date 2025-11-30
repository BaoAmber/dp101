import torch
from train import train_model
from evaluate import evaluate_image
from data_loader import get_data_loaders


def main():
    """主程序入口"""
    print("=" * 60)
    print("MNIST 手写数字识别系统")
    print("=" * 60)
    
    # 检查是否有 GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}\n")
    
    # 获取测试数据集（用于评估）
    _, _, test_dataset = get_data_loaders()
    
    # 训练阶段
    print("=" * 60)
    print("训练阶段")
    print("=" * 60)
    
    while True:
        try:
            epochs = int(input("请输入训练轮数 (epochs，输入 0 跳过训练): "))
            if epochs < 0:
                print("请输入非负整数！")
                continue
            break
        except ValueError:
            print("请输入有效的数字！")
            continue
    
    model = None
    if epochs > 0:
        model, _, test_loader, test_dataset = train_model(epochs, device)
        
        # 保存模型
        model_path = 'mnist_model.pth'
        torch.save(model.state_dict(), model_path)
        print(f"模型已保存到: {model_path}\n")
    else:
        # 尝试加载已保存的模型
        try:
            from model import SimpleNet
            model = SimpleNet().to(device)
            model.load_state_dict(torch.load('mnist_model.pth', map_location=device))
            model.eval()
            print("已加载已保存的模型。\n")
        except FileNotFoundError:
            print("错误：未找到已保存的模型，请先训练模型！")
            return
    
    # 评估阶段
    print("=" * 60)
    print("评估模式")
    print("=" * 60)
    print(f"测试集大小: {len(test_dataset)}")
    print("输入图片索引 (0-9999) 查看预测结果")
    print("输入 'q' 或 'quit' 退出程序\n")
    
    while True:
        user_input = input("请输入图片索引: ").strip().lower()
        
        if user_input in ['q', 'quit', 'exit']:
            print("程序退出。")
            break
        
        try:
            index = int(user_input)
            evaluate_image(model, test_dataset, index, device)
            print()  # 空行分隔
        except ValueError:
            print("请输入有效的数字索引或 'q' 退出！\n")
        except KeyboardInterrupt:
            print("\n\n程序被用户中断。")
            break
        except Exception as e:
            print(f"发生错误: {e}\n")


if __name__ == '__main__':
    main()

