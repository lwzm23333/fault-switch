# models/layers.py
import torch
import torch.nn as nn

class SEGating(nn.Module):
    def __init__(self, d_model: int, ratio: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model//ratio)
        self.fc2 = nn.Linear(d_model//ratio, d_model)
        self.act = nn.ReLU()
        self.sig = nn.Sigmoid()
    def forward(self, x):
        # x: (B, L, D)
        s = x.mean(dim=1)  # 全局 token 池化
        g = self.fc2(self.act(self.fc1(s)))
        g = self.sig(g).unsqueeze(1)              # (B,1,D)
        return x * g
