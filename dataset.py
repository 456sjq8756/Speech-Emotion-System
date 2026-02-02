import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.model_selection import train_test_split


class EmotionDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # 1. 获取数据 (Time, Features)
        features = self.X[idx]
        label = self.y[idx]

        # 2. 转换为 Tensor
        # CNN 需要 (Channels, Time)，所以这里需要转置一下
        # 原始: (300, 40) -> 转置后: (40, 300)
        x_tensor = torch.tensor(features, dtype=torch.float32).transpose(0, 1)
        y_tensor = torch.tensor(label, dtype=torch.long)

        return x_tensor, y_tensor


def load_data():
    """
    加载并划分训练集/测试集
    """
    try:
        X = np.load("X.npy")
        y = np.load("y.npy")

        # 划分训练集和测试集 (8:2)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        train_dataset = EmotionDataset(X_train, y_train)
        test_dataset = EmotionDataset(X_test, y_test)

        return train_dataset, test_dataset
    except FileNotFoundError:
        print("❌ 错误：找不到 X.npy 或 y.npy。请先运行 make_data.py")
        return None, None
