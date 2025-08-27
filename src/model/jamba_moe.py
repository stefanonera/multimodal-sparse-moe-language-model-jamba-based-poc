from typing import Dict, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from transformers import AutoTokenizer
from einops import reduce

from .fusion import ConcatProjectFusion
from .gating import MoELayer


class TextStubEncoder(nn.Module):
    """
    Text stub encoder (frozen)
    """

    def __init__(self, tokenizer_name: str, hidden_size: int):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self.vocab_size = self.tokenizer.vocab_size
        self.hidden_size = hidden_size
        self.emb = nn.Embedding(self.vocab_size, hidden_size)
        # Freeze
        for p in self.parameters():
            p.requires_grad = False

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        input_ids: [B, T]
        attention_mask: [B, T]
        returns pooled text embedding: [B, D]
        """
        x = self.emb(input_ids)  # [B, T, D]
        # mean pool over tokens with attention_mask
        mask = attention_mask.unsqueeze(-1).float()  # [B, T, 1]
        x = (x * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-9)  # [B, D]
        return x


class ImageEncoder(nn.Module):
    """
    Image encoder (frozen)
    """

    def __init__(self, out_dim: int):
        super().__init__()
        # small, available everywhere
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        feat_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.proj = nn.Linear(feat_dim, out_dim)
        # freeze
        for p in self.backbone.parameters():
            p.requires_grad = False
        # since it's a POC project, freeze fully
        for p in self.proj.parameters():
            p.requires_grad = False

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        pixel_values: [B, 3, H, W]
        returns: [B, out_dim]
        """
        feats = self.backbone(pixel_values)  # [B, feat_dim]
        return self.proj(feats)  # [B, out_dim]


class JambaMoEForCIFAR10(nn.Module):
    """
    Full POC model
    """

    def __init__(
        self,
        num_classes: int = 10,
        txt_hidden: int = 2048,
        img_hidden: int = 2048,
        model_dim: int = 2048,
        ffn_dim: int = 4096,
        num_experts: int = 4,
        num_layers: int = 1,
        capacity_factor: float = 1.25,
        tokenizer_name: str = "ai21labs/Jamba-v0.1",
    ):
        super().__init__()
        # encoders (frozen)
        self.txt_enc = TextStubEncoder(
            tokenizer_name=tokenizer_name, hidden_size=txt_hidden
        )
        self.img_enc = ImageEncoder(out_dim=img_hidden)

        # fusion
        self.fusion = ConcatProjectFusion(
            txt_dim=txt_hidden, img_dim=img_hidden, model_dim=model_dim
        )

        # MoE stack
        self.moe_layers = nn.ModuleList(
            [
                MoELayer(
                    d_model=model_dim,
                    d_ff=ffn_dim,
                    num_experts=num_experts,
                    capacity_factor=capacity_factor,
                    activation="swiglu",
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(model_dim)

        # classifier
        self.head = nn.Linear(model_dim, num_classes)

    @torch.no_grad()
    def freeze_encoders(self):
        for p in self.txt_enc.parameters():
            p.requires_grad = False
        for p in self.img_enc.parameters():
            p.requires_grad = False

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Returns (loss, logs) when labels provided, else (logits, logs)
        logs includes routing metrics aggregated over MoE layers.
        """
        # frozen encoders
        txt = self.txt_enc(
            input_ids=input_ids, attention_mask=attention_mask
        )  # [B, D_txt]
        img = self.img_enc(pixel_values=pixel_values)  # [B, D_img]

        # fusion to model dim
        x = self.fusion(txt, img)  # [B, D_model]

        # MoE stack with residuals
        routing_logs: List[Dict] = []
        for layer in self.moe_layers:
            x, stats = layer(x)
            routing_logs.append(stats)

        x = self.norm(x)
        logits = self.head(x)  # [B, C]

        outputs = {"logits": logits, "routing_logs": routing_logs}

        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            outputs["loss"] = loss
            return loss, outputs
        return logits, outputs
