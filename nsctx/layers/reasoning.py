"""Neuro-symbolic reasoning utilities."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuroSymbolicReasoner(nn.Module):
    """Combines neural logits with symbolic rule satisfaction."""

    def __init__(self, d_model: int, n_outputs: int):
        super().__init__()
        self.decoder = nn.Linear(d_model, n_outputs)
        self.rule_matrix = nn.Parameter(torch.randn(n_outputs, n_outputs))

    def forward(self, hub: torch.Tensor) -> torch.Tensor:
        logits = self.decoder(hub)
        rule_logits = hub @ self.rule_matrix
        return logits + 0.1 * rule_logits

    def rule_loss(self, preds: torch.Tensor) -> torch.Tensor:
        probs = preds.sigmoid()
        rule_consistency = torch.mean(probs * (1 - probs))
        return rule_consistency


class ElasticWeightConsolidation:
    """Stores Fisher diagonals and applies EWC penalty."""

    def __init__(self, lambda_: float = 0.001):
        self.lambda_ = lambda_
        self.params = None
        self.fisher = None

    def update(self, model: nn.Module, logits: torch.Tensor) -> None:
        loss = (logits ** 2).mean()
        grads = torch.autograd.grad(loss, model.parameters(), retain_graph=True, allow_unused=True)
        self.params = [p.detach().clone() for p in model.parameters()]
        self.fisher = [g.detach() ** 2 if g is not None else torch.zeros_like(p) for g, p in zip(grads, model.parameters())]

    def penalty(self, model: nn.Module) -> torch.Tensor:
        if self.params is None or self.fisher is None:
            return torch.tensor(0.0, device=next(model.parameters()).device)
        penalty = 0.0
        for p, p_old, f in zip(model.parameters(), self.params, self.fisher):
            penalty = penalty + torch.sum(f * (p - p_old) ** 2)
        return self.lambda_ * penalty
