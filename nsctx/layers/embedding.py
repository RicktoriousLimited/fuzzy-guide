"""Embedding layers for NSCTX."""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class ModalityTextEncoder(nn.Module):
    """Simple text encoder with token + positional embeddings."""

    def __init__(self, vocab_size: int, d_model: int, max_len: int = 64):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.position_embed = nn.Embedding(max_len, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(tokens.size(1), device=tokens.device)
        pos_emb = self.position_embed(positions)[None, :, :]
        tok_emb = self.token_embed(tokens)
        return self.layer_norm(tok_emb + pos_emb)


class SharedProjection(nn.Module):
    """Project modality encoders into a shared space."""

    def __init__(self, d_model: int, d_fused: int):
        super().__init__()
        self.proj = nn.Linear(d_model, d_fused)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class ModalityPooler(nn.Module):
    """Average pool a sequence representation."""

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            return x.mean(dim=1)
        weights = mask.float()
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        return torch.einsum("bn,bnd->bd", weights, x)


def build_vocab(vocab: Dict[str, int]) -> Dict[str, int]:
    """Add special tokens if needed."""

    vocab = dict(vocab)
    vocab.setdefault("<unk>", len(vocab))
    vocab.setdefault("<pad>", len(vocab))
    return vocab
