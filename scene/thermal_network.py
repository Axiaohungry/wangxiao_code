# D:\PycharmProjects\wangxiao_code\scene\thermal_network.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ThermalAttrNet(nn.Module):
    def __init__(self, input_ch=8, W=16, D=3):
        """
        热场网络 v1 (Physics Informed)
        input_ch: 8
           - 3 (XYZ)
           - 3 (Normal X,Y,Z)
           - 1 (Height)
           - 1 (Slope)
        W: 16 (保持轻量，防止显存爆炸)
        """
        super().__init__()
        self.layers = nn.ModuleList()
        # 输入层
        self.layers.append(nn.Linear(input_ch, W))

        # 隐藏层
        for i in range(D):
            self.layers.append(nn.Linear(W, W))

        # 输出层
        self.out = nn.Linear(W, 1)

    def forward(self, x):
        # x: [N, 8]
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            h = F.relu(h)

        out = self.out(h)
        return torch.sigmoid(out)