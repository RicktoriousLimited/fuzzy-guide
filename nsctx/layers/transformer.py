"""Contextual transformer layer stack."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class RoleAwareAttention(nn.Module):
    """Multi-head attention augmented with learned role embeddings."""

    def __init__(self, d_model: int, n_heads: int, n_roles: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.role_q = nn.Embedding(n_roles, d_model)
        self.role_k = nn.Embedding(n_roles, d_model)

    def forward(
        self,
        x: torch.Tensor,
        roles: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q_bias = self.role_q(roles)
        k_bias = self.role_k(roles)
        q = x + q_bias
        k = x + k_bias
        out, _ = self.attn(q, k, x, attn_mask=attn_mask)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int, n_roles: int):
        super().__init__()
        self.attn = RoleAwareAttention(d_model, n_heads, n_roles)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, roles: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x, roles)
        x = self.norm1(x + attn_out)
        ff = self.ffn(x)
        return self.norm2(x + ff)


class TransformerStack(nn.Module):
    def __init__(self, depth: int, d_model: int, n_heads: int, dim_feedforward: int, n_roles: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, dim_feedforward, n_roles) for _ in range(depth)]
        )

    def forward(self, x: torch.Tensor, roles: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, roles)
        return x
