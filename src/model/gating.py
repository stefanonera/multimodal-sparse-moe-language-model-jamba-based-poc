from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def entropy(p: torch.Tensor, dim: int = -1, eps: float = 1e-9) -> torch.Tensor:
    p = p.clamp_min(eps)
    return -(p * p.log()).sum(dim=dim)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w12 = nn.Linear(d_model, 2 * d_ff, bias=False)
        self.proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        u, v = self.w12(x).chunk(2, dim=-1)
        return self.proj(F.silu(v) * u)


class Expert(nn.Module):
    def __init__(self, d_model: int, d_ff: int, activation: str = "swiglu"):
        super().__init__()
        if activation.lower() != "swiglu":
            raise ValueError("This POC only implements SwiGLU experts.")
        self.ff = SwiGLU(d_model, d_ff)

    def forward(self, x):
        return self.ff(x)


class Top2Router(nn.Module):
    def __init__(self, d_model: int, num_experts: int, capacity_factor: float = 1.25):
        super().__init__()
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.router = nn.Linear(d_model, num_experts, bias=False)

    @torch.no_grad()
    def _balance_score(self, probs: torch.Tensor) -> torch.Tensor:
        # probs: [B, E] after softmax (over experts)
        mean_p = probs.mean(dim=0)  # [E]
        mean_p = mean_p / (mean_p.sum() + 1e-9)
        h = entropy(mean_p, dim=0)
        # normalize by log(E) to [0,1]
        return h / (torch.log(torch.tensor(self.num_experts, device=probs.device)))

    def forward(
        self, x: torch.Tensor, experts: nn.ModuleList
    ) -> Tuple[torch.Tensor, Dict]:
        """
        x: [B, D] tokens (one token per example in this POC)
        experts: list of Expert modules
        """
        B, D = x.shape
        E = self.num_experts

        logits = self.router(x)  # [B, E]
        probs = F.softmax(logits, dim=-1)  # [B, E]

        # top-2 indices and weights per token
        gate_weights, topk_idx = probs.topk(k=2, dim=-1)  # [B, 2], [B, 2]
        w1, w2 = gate_weights.unbind(dim=-1)  # [B], [B]
        e1, e2 = topk_idx.unbind(dim=-1)  # [B], [B]

        # capacity per expert
        # total slots = ceil(capacity_factor * (B * 2 / E))
        cap = int(self.capacity_factor * (B * 2 / E) + 1e-9)
        cap = max(1, cap)

        # build assignment lists per expert with scores
        device = x.device
        expert_inputs = [[] for _ in range(E)]
        expert_weights = [[] for _ in range(E)]
        expert_token_idx = [[] for _ in range(E)]

        # primary and secondary routes
        for t in range(B):
            for k, e_idx, w in [
                (
                    0,
                    int(e1[t].item()),
                    w1[t],
                ),  # Use .item() for index, keep tensor for weight
                (
                    1,
                    int(e2[t].item()),
                    w2[t],
                ),  # Use .item() for index, keep tensor for weight
            ]:
                expert_inputs[e_idx].append(x[t].unsqueeze(0))
                expert_weights[e_idx].append(w)
                expert_token_idx[e_idx].append(t)

        # for each expert, sort by weight desc and keep up to capacity
        kept_mask = torch.zeros(
            B, E, dtype=torch.bool, device=device
        )  # for heatmaps and counts
        outputs = torch.zeros_like(x)

        for e in range(E):
            if len(expert_inputs[e]) == 0:
                continue
            X_e = torch.cat(expert_inputs[e], dim=0)  # [N_e, D]
            W_e = torch.stack(
                expert_weights[e]
            )  # [N_e] - stack tensors instead of converting to tensor
            T_e = torch.tensor(expert_token_idx[e], device=device, dtype=torch.long)

            # sort by weight
            order = torch.argsort(W_e, descending=True)
            X_e = X_e[order]
            W_e = W_e[order]
            T_e = T_e[order]

            N_keep = min(cap, X_e.size(0))
            X_keep = X_e[:N_keep]
            W_keep = W_e[:N_keep]
            T_keep = T_e[:N_keep]

            # process through expert
            y_e = experts[e](X_keep)  # [N_keep, D]

            # scatter-add weighted outputs back to tokens
            outputs.index_add_(0, T_keep, y_e * W_keep.unsqueeze(-1))

            kept_mask[T_keep, e] = True

        counts_per_expert = kept_mask.sum(dim=0)  # [E]
        balance_score = (
            self._balance_score(probs).detach().item()
        )  # Add .detach() before .item()
        topk_dist = gate_weights / (gate_weights.sum(dim=-1, keepdim=True) + 1e-9)
        topk_ent = (
            entropy(topk_dist, dim=-1).mean().detach().item()
        )  # Add .detach() before .item()

        stats = {
            "counts_per_expert": counts_per_expert.detach().cpu(),  # tensor [E]
            "balance_score": balance_score,
            "topk_entropy": topk_ent,
            "kept_mask": kept_mask.detach().cpu(),  # [B, E] for heatmaps
        }
        return outputs, stats


class MoELayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int,
        capacity_factor: float = 1.25,
        activation: str = "swiglu",
    ):
        super().__init__()
        self.router = Top2Router(d_model, num_experts, capacity_factor)
        self.experts = nn.ModuleList(
            [Expert(d_model, d_ff, activation=activation) for _ in range(num_experts)]
        )
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        # x: [B, D]
        x_norm = self.ln(x)
        y, stats = self.router(x_norm, self.experts)
        # residual
        return x + y, stats
