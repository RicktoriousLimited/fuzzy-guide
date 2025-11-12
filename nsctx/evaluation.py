"""Evaluation utilities for NSCTX."""
from __future__ import annotations

from typing import Dict

import torch

from .model import NSCTXModel


class Evaluator:
    def __init__(self, model: NSCTXModel, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device

    def predict(self, tokens: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            mask = (tokens != 0).float()
            batch = {"text": tokens.to(self.device), "mask": mask.to(self.device)}
            outputs = self.model(batch)
            return outputs["logits"].sigmoid()

    def explain(self, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.model.eval()
        with torch.no_grad():
            mask = (tokens != 0).float()
            batch = {"text": tokens.to(self.device), "mask": mask.to(self.device)}
            outputs = self.model(batch)
            return outputs
