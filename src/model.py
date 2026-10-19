"""AGRU 模型实现 (对齐研报图1: Linear→Tanh→GRU→Attn→Mul/Sum/Concat→Linear→Output)"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(hidden_dim) * 0.01)
        self.scale = hidden_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(x, self.query) * self.scale
        weights = F.softmax(scores, dim=1)
        return torch.bmm(weights.unsqueeze(1), x).squeeze(1)


class AGRU(nn.Module):
    """
    结构:
      输入 (B, T, input_dim)
        → Linear + Tanh          (B, T, hidden)
        → GRU 2层 hidden=64      (B, T, hidden)
        → 取末步 gru_last        (B, hidden)  [旁路]
        → Attention(gru_seq)     (B, hidden)
        → 融合 Mul/Sum/Concat    (B, 4*hidden)
        → Dropout → Linear → 三分类
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_gru_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
        )
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_gru_layers,
            batch_first=True,
            dropout=dropout if num_gru_layers > 1 else 0.0,
        )
        self.attention = FeatureAttention(hidden_dim)

        self.fusion_dim = hidden_dim * 4
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        gru_out, _ = self.gru(encoded)

        gru_last = gru_out[:, -1, :]
        attn_out = self.attention(gru_out)

        mul = gru_last * attn_out
        sum_ = gru_last + attn_out
        concat = torch.cat([gru_last, attn_out], dim=-1)

        fused = torch.cat([mul, sum_, concat], dim=-1)
        logits = self.classifier(self.dropout(fused))
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(x), dim=-1)
