import numpy as np
from sklearn.metrics import confusion_matrix  # ✨ 新增：用于计算混淆矩阵
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import load_data
from model import CNN_LSTM_Model, CNN_Model, LSTM_Model
import os
import csv  # ✨ 新增：用于保存训练日志

# ==========================================
# ✨ 调参修改 1：增加学习时间、批次大小和脑容量
# ==========================================
BATCH_SIZE = 32         # 原来是 16，增加到 32 让梯度下降更稳定
LEARNING_RATE = 0.001
EPOCHS = 100            # 原来是 50，增加到 100 让模型学得更透彻
NUM_CLASSES = 6
INPUT_SIZE = 40
HIDDEN_SIZE = 256       # 原来是 128，增加 LSTM 脑容量

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"当前使用的训练设备: {DEVICE}")


def train():
    print("正在加载数据集...")
    train_dataset, test_dataset = load_data()

    if train_dataset is None:
        print("❌ 数据加载失败，请检查 make_data.py 是否运行成功")
        return

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"训练集数量: {len(train_dataset)}, 测试集数量: {len(test_dataset)}\n")

    models_to_train = {
        "CNN-LSTM (本项目主模型)": (CNN_LSTM_Model, 'best_model.pth'),
        "纯 CNN (对比基线模型)": (CNN_Model, 'cnn_model.pth'),
        "纯 LSTM (对比基线模型)": (LSTM_Model, 'lstm_model.pth')
    }

    for model_name, (ModelClass, save_path) in models_to_train.items():
        print(f"{'=' * 50}")
        print(f"🚀 开始训练: {model_name}")
        print(f"{'=' * 50}")

        model = ModelClass(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_classes=NUM_CLASSES)
        model = model.to(DEVICE)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # ==========================================
        # ✨ 调参修改 3 (上)：定义学习率调度器
        # 每隔 30 个 Epoch，把学习率乘以 0.5 (即减半)
        # ==========================================
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

        best_acc = 0.0

        # 用于记录当前模型的训练历史
        training_history = []
        csv_filename = save_path.replace('.pth', '_log.csv')

        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0

            for features, labels in train_loader:
                features, labels = features.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for features, labels in test_loader:
                    features, labels = features.to(DEVICE), labels.to(DEVICE)
                    outputs = model(features)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            epoch_acc = 100 * correct / total
            avg_loss = running_loss / len(train_loader)

            # 获取当前最新的学习率，打印出来让我们看到它在变小
            current_lr = scheduler.get_last_lr()[0]
            print(f"[{model_name}] Epoch [{epoch + 1}/{EPOCHS}] -> Loss: {avg_loss:.4f} | Accuracy: {epoch_acc:.2f}% | LR: {current_lr:.6f}")

            # 记录当前 Epoch 的数据
            training_history.append([epoch + 1, avg_loss, epoch_acc])

            if epoch_acc > best_acc:
                best_acc = epoch_acc
                # 1. 保存模型权重
                torch.save(model.state_dict(), save_path)

                # ✨ 新增：2. 计算并保存混淆矩阵 (为了让网页端能显示)
                all_preds = []
                all_labels = []
                model.eval()
                with torch.no_grad():
                    for features, labels in test_loader:
                        features = features.to(DEVICE)
                        outputs = model(features)
                        _, predicted = torch.max(outputs.data, 1)
                        all_preds.extend(predicted.cpu().numpy())
                        all_labels.extend(labels.numpy())

                # 计算矩阵
                cm = confusion_matrix(all_labels, all_preds)
                # 保存为 .npy 文件，方便 app.py 读取
                cm_filename = save_path.replace('.pth', '_cm.npy')
                np.save(cm_filename, cm)

                print(f"  🔥 准确率提升！已保存最优模型和混淆矩阵")

            # ==========================================
            # ✨ 调参修改 3 (下)：更新学习率
            # 注意：这句必须在每个 epoch 循环的最后执行！
            # ==========================================
            scheduler.step()

        # 将训练记录写入 CSV 文件
        with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Loss', 'Accuracy'])
            writer.writerows(training_history)
        print(f"💾 {model_name} 的训练日志已保存至 {csv_filename}")

    print("\n🎉 所有对比实验模型及其训练日志均已保存完毕！")


if __name__ == '__main__':
    train()
