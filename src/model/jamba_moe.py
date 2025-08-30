from typing import Dict, Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from transformers import AutoTokenizer

from .fusion import ConcatProjectFusion
from .gating import MoELayer


class TextStubEncoder(nn.Module):
    """
    Text stub encoder (frozen)
    """

    def __init__(self, tokenizer_name: str, hidden_size: int):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, use_fast=True, legacy=False
        )
        self.vocab_size = self.tokenizer.vocab_size
        self.hidden_size = hidden_size
        self.emb = nn.Embedding(self.vocab_size, hidden_size)
        for p in self.parameters():
            p.requires_grad = False

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        x = self.emb(input_ids)  # [B, T, D]
        mask = attention_mask.unsqueeze(-1).float()
        x = (x * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-9)  # [B, D]
        return x


class ImageEncoder(nn.Module):
    """
    Image encoder (frozen)
    """

    def __init__(self, out_dim: int, frozen: bool = True):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        feat_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.proj = nn.Linear(feat_dim, out_dim)
        if frozen:
            for p in self.backbone.parameters():
                p.requires_grad = False
            for p in self.proj.parameters():
                p.requires_grad = False

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(pixel_values)  # [B, feat_dim]
        return self.proj(feats)  # [B, out_dim]


class VideoEncoder(nn.Module):
    """
    Video encoder (frozen)
    Apply resnet18 per frame, mean-pool over time, then proj.
    Input: [B, F, 3, H, W]
    """

    def __init__(self, out_dim: int, frozen: bool = True):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        feat_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.proj = nn.Linear(feat_dim, out_dim)
        if frozen:
            for p in self.backbone.parameters():
                p.requires_grad = False
            for p in self.proj.parameters():
                p.requires_grad = False

    def forward(self, video_frames: torch.Tensor) -> torch.Tensor:
        # video_frames: [B, F, 3, H, W]
        B, F, C, H, W = video_frames.shape
        x = video_frames.reshape(B * F, C, H, W)  # [B*F, 3, H, W]
        feats = self.backbone(x)  # [B*F, feat_dim]
        feats = feats.view(B, F, -1).mean(dim=1)  # [B, feat_dim]
        return self.proj(feats)  # [B, out_dim]


class JambaMoEClassifier(nn.Module):
    """
    General multimodal MoE classifier
    """

    def __init__(
        self,
        num_classes: int,
        active_modalities: List[str],  # subset of ["text","image","video"]
        txt_hidden: int = 1024,
        img_hidden: int = 1024,
        vid_hidden: int = 1024,
        model_dim: int = 1024,
        ffn_dim: int = 2048,
        num_experts: int = 4,
        num_layers: int = 1,
        capacity_factor: float = 1.25,
        tokenizer_name: str = "ai21labs/Jamba-v0.1",
        freeze_encoders: bool = True,
    ):
        super().__init__()
        self.active_modalities = set(active_modalities)

        # encoders
        if "text" in self.active_modalities:
            self.txt_enc = TextStubEncoder(
                tokenizer_name=tokenizer_name, hidden_size=txt_hidden
            )
        else:
            self.txt_enc = None

        if "image" in self.active_modalities:
            self.img_enc = ImageEncoder(out_dim=img_hidden, frozen=freeze_encoders)
        else:
            self.img_enc = None

        if "video" in self.active_modalities:
            self.vid_enc = VideoEncoder(out_dim=vid_hidden, frozen=freeze_encoders)
        else:
            self.vid_enc = None

        # fusion: concat all present embeddings -> model_dim
        in_dim = (
            (txt_hidden if self.txt_enc else 0)
            + (img_hidden if self.img_enc else 0)
            + (vid_hidden if self.vid_enc else 0)
        )
        self.fusion = (
            ConcatProjectFusion(txt_dim=in_dim, img_dim=0, model_dim=model_dim)
            if in_dim != model_dim
            else nn.LayerNorm(model_dim)
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
        for enc in [self.txt_enc, self.img_enc, self.vid_enc]:
            if enc is None:
                continue
            for p in enc.parameters():
                p.requires_grad = False

    def _collect_embeddings(
        self, pixel_values, video_frames, input_ids, attention_mask
    ):
        device = next(self.parameters()).device
        embs = []
        if (
            self.txt_enc is not None
            and input_ids is not None
            and attention_mask is not None
        ):
            txt_emb = self.txt_enc(input_ids=input_ids, attention_mask=attention_mask)
            embs.append(txt_emb.to(device))  # [B, D_txt]
        if self.img_enc is not None and pixel_values is not None:
            img_emb = self.img_enc(pixel_values=pixel_values)
            embs.append(img_emb.to(device))  # [B, D_img]
        if self.vid_enc is not None and video_frames is not None:
            vid_emb = self.vid_enc(video_frames=video_frames)
            embs.append(vid_emb.to(device))  # [B, D_vid]
        if len(embs) == 0:
            raise ValueError("No modalities provided to the model forward.")
        return torch.cat(embs, dim=-1) if len(embs) > 1 else embs[0]

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        video_frames: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        # get the device that this model is on
        device = next(self.parameters()).device

        x = self._collect_embeddings(
            pixel_values, video_frames, input_ids, attention_mask
        )  # [B, D_in]

        # ensure x is on the correct device
        x = x.to(device)

        if isinstance(self.fusion, ConcatProjectFusion):
            x = self.fusion(
                x,
                torch.empty(
                    0, device=device
                ),  # make sure empty tensor is on same device
            )  # hack: ConcatProjectFusion expects 2 args; we use only first
        else:
            x = self.fusion(x)

        routing_logs: List[Dict] = []
        for layer in self.moe_layers:
            x, stats = layer(x)
            routing_logs.append(stats)

        x = self.norm(x)
        logits = self.head(x)

        outputs = {"logits": logits, "routing_logs": routing_logs}
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            outputs["loss"] = loss
            return loss, outputs
        return logits, outputs


# backward-compat
class JambaMoEForCIFAR10(JambaMoEClassifier):
    def __init__(self, **kw):
        super().__init__(num_classes=10, active_modalities=["text", "image"], **kw)
