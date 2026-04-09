# training/contrastive.py
# 评估损失

import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.tau = temperature

    def forward(self, z, pos_idx):
        """
        z: (B, D) — batch 内 embedding
        pos_idx: (B,) — 每个样本的“正对”索引（同工况不同噪声）
        """
        B, D = z.shape
        z = F.normalize(z, dim=1)
        sim = z @ z.t() / self.tau          # (B,B)
        mask = torch.eye(B, device=z.device).bool()
        sim.masked_fill_(mask, float('-inf'))
        targets = pos_idx
        loss = F.cross_entropy(sim, targets)
        return loss
