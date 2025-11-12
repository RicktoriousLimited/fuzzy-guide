"""Data utilities for NSCTX."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import torch


@dataclass
class Sample:
    """Represents a single training example."""

    text: str
    label: torch.Tensor
    modalities: Dict[str, torch.Tensor]


class InMemoryDataset(torch.utils.data.Dataset):
    """A tiny dataset living entirely in memory.

    This keeps the app self-contained while still exercising the training
    pipeline end-to-end.
    """

    def __init__(self, samples: Sequence[Sample]):
        self._samples = list(samples)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._samples)

    def __getitem__(self, idx: int) -> Sample:
        return self._samples[idx]


def build_demo_dataset(vocab: Iterable[str]) -> InMemoryDataset:
    """Construct a simple dataset with toy multi-modal targets.

    The labels are represented as three binary attributes that loosely map to
    sentiment, responsibility, and physical state cues.
    """

    vocab_to_idx = {token: i for i, token in enumerate(vocab)}

    def encode_text(text: str) -> torch.Tensor:
        indices = [vocab_to_idx.get(tok, vocab_to_idx["<unk>"]) for tok in text.split()]
        return torch.tensor(indices, dtype=torch.long)

    samples: List[Sample] = []
    labels = {
        "the boy kicked the ball": torch.tensor([1.0, 0.0, 0.0]),
        "the boy apologized": torch.tensor([0.0, 1.0, 0.0]),
        "the man was angry": torch.tensor([1.0, 1.0, 0.0]),
        "the phone was broken": torch.tensor([0.0, 0.0, 1.0]),
    }
    for text, label in labels.items():
        samples.append(
            Sample(
                text=text,
                label=label,
                modalities={"text": encode_text(text)},
            )
        )
    return InMemoryDataset(samples)
