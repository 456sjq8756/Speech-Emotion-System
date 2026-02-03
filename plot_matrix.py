import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from dataset import load_data
from model import CNN_LSTM_Model

# --- 1. 配置参数 (需与训练时一致) ---
INPUT_SIZE = 40
HIDDEN_SIZE = 128
NUM_CLASSES = 6
BATCH_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 标签列表 (对应 make_data.py 的顺序)
# 0:angry, 1:fear, 2:happy, 3:neutral, 4:sad, 5:surprise
LABELS = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


def plot_confusion_matrix():
    print("正在加载测试数据...")
    _, test_dataset = load_data()
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("正在加载模型...")
    model = CNN_LSTM_Model(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_classes=NUM_CLASSES)
    # 加载你训练好的权重
    model.load_state_dict(torch.load('best_model.pth', map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    y_true = []
    y_pred = []

    print("正在进行推理计算...")
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(DEVICE)
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)

            # 收集真实标签和预测标签
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # --- 2. 计算混淆矩阵 ---
    cm = confusion_matrix(y_true, y_pred)

    # --- 3. 绘制热力图 ---
    plt.figure(figsize=(10, 8))

    # 绘制热力图
    # annot=True: 显示数值
    # fmt='d': 数值格式为整数
    # cmap='Blues': 颜色主题 (可选 'OrRd', 'Greens' 等)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=LABELS, yticklabels=LABELS)

    plt.title('Confusion Matrix of Speech Emotion Recognition', fontsize=15)
    plt.ylabel('True Label (真实标签)', fontsize=12)
    plt.xlabel('Predicted Label (预测标签)', fontsize=12)

    # 保存图片
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("✅ 绘图完成！已保存为 'confusion_matrix.png'")
    plt.show()


if __name__ == '__main__':
    plot_confusion_matrix()
