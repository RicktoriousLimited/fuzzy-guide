"""Meta-learning utilities."""
from __future__ import annotations

import torch
import torch.nn as nn


class MetaLearner(nn.Module):
    """Performs a single inner-loop adaptation step."""

    def __init__(self, base_model: nn.Module, lr: float = 1e-3):
        super().__init__()
        self.base_model = base_model
        self.lr = lr

    def forward(self, loss: torch.Tensor) -> None:
        grads = torch.autograd.grad(loss, self.base_model.parameters(), retain_graph=True, allow_unused=True)
        for param, grad in zip(self.base_model.parameters(), grads):
            if grad is None:
                continue
            param.data = param.data - self.lr * grad
