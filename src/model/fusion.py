from typing import Tuple
import torch
import torch.nn as nn


class ConcatProjectFusion(nn.Module):
    def __init__(self, txt_dim: int, img_dim: int, model_dim: int):
        super().__init__()
        self.proj = nn.Linear(txt_dim + img_dim, model_dim)
        self.ln = nn.LayerNorm(model_dim)

    def forward(self, txt_emb: torch.Tensor, img_emb: torch.Tensor) -> torch.Tensor:
        """
        txt_emb: [B, T_txt, D_txt] pooled to [B, D_txt] before calling
        img_emb: [B, D_img]
        returns: [B, D_model]
        """
        x = torch.cat([txt_emb, img_emb], dim=-1)
        x = self.proj(x)
        return self.ln(x)
