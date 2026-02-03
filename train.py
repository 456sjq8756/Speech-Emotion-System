import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import load_data
from model import CNN_LSTM_Model
import os

# --- 1. 超参数配置 ---
BATCH_SIZE = 16  # 每一批训练多少个样本 (电脑卡就改小点，比如 8)
LEARNING_RATE = 0.001  # 学习率
EPOCHS = 50  # 训练轮数 (先跑50轮试试)
NUM_CLASSES = 6  # 对应 make_data.py 里的 6 种情感
INPUT_SIZE = 40  # MFCC 特征维度

# 检查是否有显卡 (你之前测试是 CPU，这里会自动适配)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"当前使用的训练设备: {DEVICE}")


def train():
    # --- 2. 加载数据 ---
    print("正在加载数据集...")
    train_dataset, test_dataset = load_data()

    if train_dataset is None:
        print("❌ 数据加载失败，请检查 make_data.py 是否运行成功")
        return

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"训练集数量: {len(train_dataset)}, 测试集数量: {len(test_dataset)}")

    # --- 3. 初始化模型 ---
    model = CNN_LSTM_Model(input_size=INPUT_SIZE, hidden_size=128, num_classes=NUM_CLASSES)
    model = model.to(DEVICE)  # 搬到 GPU/CPU

    # --- 4. 定义损失函数和优化器 ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 记录最佳准确率
    best_acc = 0.0

    # --- 5. 开始训练循环 ---
    for epoch in range(EPOCHS):
        model.train()  # 切换到训练模式
        running_loss = 0.0

        for features, labels in train_loader:
            features, labels = features.to(DEVICE), labels.to(DEVICE)

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            outputs = model(features)

            # 计算损失
            loss = criterion(outputs, labels)

            # 反向传播与优化
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # --- 6. 每个 Epoch 结束后进行测试 (验证) ---
        model.eval()  # 切换到评估模式
        correct = 0
        total = 0
        with torch.no_grad():  # 测试时不需要计算梯度，省内存
            for features, labels in test_loader:
                features, labels = features.to(DEVICE), labels.to(DEVICE)
                outputs = model(features)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_acc = 100 * correct / total
        avg_loss = running_loss / len(train_loader)

        print(f"Epoch [{epoch + 1}/{EPOCHS}] -> Loss: {avg_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

        # 保存最佳模型
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"  🔥 准确率提升！模型已保存为 best_model.pth")

    print("\n训练结束！")
    print(f"最高准确率: {best_acc:.2f}%")
    print("请使用 'best_model.pth' 进行后续的可视化展示。")


if __name__ == '__main__':
    train()
