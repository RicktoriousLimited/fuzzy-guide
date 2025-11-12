"""Training utilities for NSCTX."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch.utils.data import DataLoader

from .data import InMemoryDataset
from .model import NSCTXModel


@dataclass
class TrainingConfig:
    epochs: int = 10
    batch_size: int = 2
    lr: float = 1e-3
    device: str = "cpu"


class Trainer:
    """Simple trainer implementing the multi-loss objective."""

    def __init__(self, model: NSCTXModel, config: TrainingConfig):
        self.model = model.to(config.device)
        self.config = config
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)

    def _collate(self, batch):
        tokens = torch.nn.utils.rnn.pad_sequence(
            [sample.modalities["text"] for sample in batch], batch_first=True, padding_value=0
        )
        mask = (tokens != 0).float()
        labels = torch.stack([sample.label for sample in batch])
        return {"text": tokens.to(self.config.device), "mask": mask.to(self.config.device)}, labels.to(self.config.device)

    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        metrics = {"lm": 0.0, "rule": 0.0, "ewc": 0.0, "consistency": 0.0, "total": 0.0}
        for batch_samples in loader:
            batch, labels = batch_samples
            self.optimizer.zero_grad()
            outputs = self.model(batch)
            self.model.ewc.update(self.model.reasoner, outputs["logits"])
            losses = self.model.compute_losses(outputs, labels)
            self.model.adapt(losses["lm"])
            losses["total"].backward()
            self.optimizer.step()
            for key in metrics:
                metrics[key] += losses[key].item()
        num_batches = len(loader)
        for key in metrics:
            metrics[key] /= max(num_batches, 1)
        return metrics

    def fit(self, dataset: InMemoryDataset) -> Dict[str, float]:
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True, collate_fn=self._collate)
        last_metrics = {}
        for _ in range(self.config.epochs):
            last_metrics = self.train_epoch(loader)
        return last_metrics
