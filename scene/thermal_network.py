# D:\PycharmProjects\wangxiao_code\scene\thermal_network.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ThermalAttrNet(nn.Module):
    def __init__(self, input_ch=3, W=16, D=3):  # <--- 必须是 16 !!!
        """
        极速版温度网络 (专门优化 6GB 显存 + 440万点云)
        W: 16 (从 64 降到 16，显存占用降低 75%)
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
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            h = F.relu(h)

        out = self.out(h)
        return torch.sigmoid(out)