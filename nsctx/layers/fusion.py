"""Cross-modal fusion utilities."""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityFusion(nn.Module):
    """Fuse modality embeddings using learned attention weights."""

    def __init__(self, d_modal: int, d_fused: int, modalities: Tuple[str, ...]):
        super().__init__()
        self.modalities = modalities
        self.projections = nn.ModuleDict({m: nn.Linear(d_modal, d_fused) for m in modalities})
        self.gate = nn.Parameter(torch.zeros(len(modalities)))

    def forward(self, modality_reprs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        pooled = []
        for m in self.modalities:
            pooled.append(self.projections[m](modality_reprs[m]))
        stacked = torch.stack(pooled, dim=1)
        weights = F.softmax(self.gate, dim=0)
        fused = (weights[None, :, None] * stacked).sum(dim=1)
        return fused, weights


class ContrastiveAlignmentLoss(nn.Module):
    """InfoNCE-style loss for aligning two modalities."""

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
        logits = a @ b.t() / self.temperature
        targets = torch.arange(a.size(0), device=a.device)
        loss_a = F.cross_entropy(logits, targets)
        loss_b = F.cross_entropy(logits.t(), targets)
        return 0.5 * (loss_a + loss_b)
