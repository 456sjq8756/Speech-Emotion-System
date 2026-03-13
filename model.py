import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    注意力机制模块：对应开题报告中提到的“强化关键情感特征的权重分配”
    """
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x shape: (batch_size, time_steps, hidden_dim)
        weights = F.softmax(self.attn(x), dim=1)  # 计算权重
        context = torch.sum(weights * x, dim=1)  # 加权求和
        return context, weights


# ==========================================
# 核心主模型：CNN-LSTM-Attention (本项目提出)
# ==========================================
class CNN_LSTM_Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(CNN_LSTM_Model, self).__init__()

        # 1. CNN 层
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.3),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.3)
        )

        # 2. LSTM 层 [cite: 1]
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_size,
                            num_layers=2, batch_first=True, bidirectional=True)

        # 3. 注意力机制
        self.attention = Attention(hidden_size * 2)

        # 4. 全连接层分类
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        attn_out, weights = self.attention(lstm_out)
        logits = self.fc(attn_out)
        return logits


# ==========================================
# 对比基线模型 1：纯 CNN 模型
# ==========================================
class CNN_Model(nn.Module):
    """用于消融实验：验证缺少 LSTM 时，时序建模能力下降的影响"""
    def __init__(self, input_size, hidden_size, num_classes):
        super(CNN_Model, self).__init__()
        # 保持与主模型完全一致的卷积结构
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.3),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.3)
        )
        # 直接接全连接层 (128 是上一层输出的通道数)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        # 对时间维度进行全局平均池化 (Global Average Pooling)
        x = torch.mean(x, dim=2)
        logits = self.fc(x)
        return logits


# ==========================================
# 对比基线模型 2：纯 LSTM 模型
# ==========================================
class LSTM_Model(nn.Module):
    """用于消融实验：验证缺少 CNN 时，局部声学特征提取能力下降的影响"""
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTM_Model, self).__init__()
        # 直接让 LSTM 处理原始输入 (MFCC 特征)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=2, batch_first=True, bidirectional=True)
        # 加上 Attention 保持公平
        self.attention = Attention(hidden_size * 2)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # 维度调整：(Batch, Features, Time) -> (Batch, Time, Features)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        attn_out, weights = self.attention(lstm_out)
        logits = self.fc(attn_out)
        return logits
