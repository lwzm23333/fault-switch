import torch
import torch.nn as nn
from models_layers import SEGating


class PhysBandBias(nn.Module):
    """物理频带偏置：根据 f_center 生成加权偏置"""
    def __init__(self, d_model: int, bands, sigma: float = 30.0):
        super().__init__()
        self.bands = bands
        self.sigma = sigma
        # 注意：bias 是标量到 d_model 的投影（输入形状 (B,L,1)）
        self.proj = nn.Linear(1, d_model)

    def forward(self, f_center: torch.Tensor):
        # f_center: (B, L, 1)
        B, L, _ = f_center.shape
        bias = torch.zeros(B, L, 1, device=f_center.device)
        for lo, hi in self.bands:
            c = (lo + hi) / 2
            bias = bias + torch.exp(-0.5 * ((f_center - c) / self.sigma) ** 2)
        return self.proj(bias)  # (B,L,D)


class IMFTokenFuser(nn.Module):
    """
    功能：基于轻量 Transformer + 物理偏置的 token 融合器
    输入:
        tokens:   (B, L) 或 (B, L, 1)
        f_center: (B, L) 或 (B, L, 1)
    输出:
        z: (B, D)    段级融合特征
        h: (B, L, D) token-level 表示
    """

    def __init__(self, token_dim=1, d_model=64, nhead=1, num_layers=2,
                 dropout=0.1, se_ratio=8, bands=((50,70),(300,340))):
        super().__init__()

        # 映射 scalar token → D 维
        self.token_proj = nn.Linear(token_dim, d_model)

        # 轻量 Transformer
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # 物理带偏置
        self.phys = PhysBandBias(d_model, bands)
        self.se = SEGating(d_model, se_ratio)
        self.norm = nn.LayerNorm(d_model)

        # CLS-like 输出映射
        self.cls = nn.Linear(d_model, d_model)

    def forward(self, tokens, f_center):
        """
        tokens: (B,L) or (B,L,1)
        f_center: (B,L) or (B,L,1)
        """
        # ---- reshape ----
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(-1)     # (B,L,1)
        if f_center.dim() == 2:
            f_center = f_center.unsqueeze(-1) # (B,L,1)

        # ---- projection to D dim ----
        tok_emb = self.token_proj(tokens)       # (B,L,D)

        # ---- add physical bias ----
        phys_bias = self.phys(f_center)         # (B,L,D)
        h = tok_emb + phys_bias

        # ---- transformer encoder ----
        h = self.encoder(h)

        # ---- SE + LN ----
        h = self.se(h)
        h = self.norm(h)

        # ---- mean pooling ----
        z = h.mean(dim=1)                       # (B,D)
        z = self.cls(z)

        return z, h
