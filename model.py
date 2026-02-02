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


class CNN_LSTM_Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(CNN_LSTM_Model, self).__init__()

        # 1. CNN 层：提取局部声学特征 (如 MFCC/Mel-spectrogram)
        # 假设输入特征维度为 (Batch, Features, Time)
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

        # 2. LSTM 层：捕捉时序动态变化 [cite: 17]
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_size,
                            num_layers=2, batch_first=True, bidirectional=True)

        # 3. 注意力机制
        self.attention = Attention(hidden_size * 2)  # 双向 LSTM 维度翻倍

        # 4. 全连接层分类
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # x shape: (Batch, Input_Size, Time_Steps) -> 适配 Conv1d

        # Pass through CNN
        x = self.cnn(x)

        # 维度调整：CNN 输出为 (Batch, Channels, Time)，LSTM 需要 (Batch, Time, Features)
        x = x.permute(0, 2, 1)

        # Pass through LSTM
        lstm_out, _ = self.lstm(x)

        # Pass through Attention
        attn_out, weights = self.attention(lstm_out)

        # Classifier
        logits = self.fc(attn_out)
        return logits
