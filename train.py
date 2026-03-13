import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import load_data
from model import CNN_LSTM_Model, CNN_Model, LSTM_Model
import os
import csv  # ✨ 新增：用于保存训练日志

# --- 1. 超参数配置 ---
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 50
NUM_CLASSES = 6
INPUT_SIZE = 40
HIDDEN_SIZE = 128

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

        best_acc = 0.0

        # ✨ 新增：用于记录当前模型的训练历史
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

            print(f"[{model_name}] Epoch [{epoch + 1}/{EPOCHS}] -> Loss: {avg_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

            # 记录当前 Epoch 的数据
            training_history.append([epoch + 1, avg_loss, epoch_acc])

            if epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), save_path)
                print(f"  🔥 准确率提升！已保存最优模型权重")

        # ✨ 新增：将训练记录写入 CSV 文件
        with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Loss', 'Accuracy'])
            writer.writerows(training_history)
        print(f"💾 {model_name} 的训练日志已保存至 {csv_filename}")

    print("\n🎉 所有对比实验模型及其训练日志均已保存完毕！")


if __name__ == '__main__':
    train()
