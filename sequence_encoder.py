import torch
import torch.nn as nn


class ChemicalEquationCNN(nn.Module):
    def __init__(self, num_chars, hidden_dim=512, gru_layers=2):
        super().__init__()

        # 专门为化学方程式设计的CNN架构
        self.cnn_features = nn.Sequential(
            # 输入: [batch, 1, 32, 128]
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [16, 64]

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [8, 32]

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # [4, 32]

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # [2, 32]

            nn.Conv2d(256, 512, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),  # [1, 31]
        )

        # BiGRU序列建模
        self.gru = nn.GRU(
            input_size=512,
            hidden_size=hidden_dim // 2,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )

        # 输出投影层
        self.output_proj = nn.Linear(hidden_dim, num_chars)

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # CNN特征提取
        cnn_features = self.cnn_features(x)  # [batch, 512, 1, 31]

        # 重塑为序列格式 [batch, sequence_length, features]
        cnn_features = cnn_features.squeeze(2)  # [batch, 512, 31]
        cnn_features = cnn_features.permute(0, 2, 1)  # [batch, 31, 512]

        # 通过BiGRU
        sequence_output, _ = self.gru(cnn_features)

        # 输出投影
        logits = self.output_proj(sequence_output)

        return logits, sequence_output