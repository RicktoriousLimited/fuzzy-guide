"""Data utilities for NSCTX."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import torch


ConversationMessage = Tuple[str, str]


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


def encode_text(text: str, vocab_to_idx: Dict[str, int]) -> torch.Tensor:
    """Convert a whitespace-tokenised string into tensor indices."""

    indices = [vocab_to_idx.get(tok, vocab_to_idx["<unk>"]) for tok in text.split()]
    return torch.tensor(indices, dtype=torch.long)


def conversation_to_text(messages: Sequence[ConversationMessage]) -> str:
    """Serialise a list of ``(role, content)`` pairs into a token sequence."""

    parts: List[str] = []
    for role, content in messages:
        parts.append(f"<{role.lower()}>")
        parts.extend(content.strip().split())
    return " ".join(parts)


def build_demo_dataset(vocab: Iterable[str]) -> InMemoryDataset:
    """Construct a conversation-style dataset with toy multi-modal targets.

    Each sample is represented as a short user/assistant dialogue that maps to
    the same three binary attributes used in the original single-utterance
    specification. This keeps the downstream training and reasoning logic the
    same while presenting the model with conversational context.
    """

    vocab_to_idx = {token: i for i, token in enumerate(vocab)}

    conversation_specs: List[Tuple[Sequence[ConversationMessage], torch.Tensor]] = [
        (
            [
                ("user", "what happened in the game"),
                ("assistant", "the boy kicked the ball"),
            ],
            torch.tensor([1.0, 0.0, 0.0]),
        ),
        (
            [
                ("user", "did the boy play again"),
                ("assistant", "the boy kicked the ball again"),
            ],
            torch.tensor([1.0, 0.0, 0.0]),
        ),
        (
            [
                ("user", "why was everyone quiet"),
                ("assistant", "the boy apologized"),
            ],
            torch.tensor([0.0, 1.0, 0.0]),
        ),
        (
            [
                ("user", "did the boy say sorry"),
                ("assistant", "the boy apologized to his friend"),
            ],
            torch.tensor([0.0, 1.0, 0.0]),
        ),
        (
            [
                ("user", "what made the man upset"),
                ("assistant", "the man was angry"),
            ],
            torch.tensor([1.0, 1.0, 0.0]),
        ),
        (
            [
                ("user", "why was the man angry"),
                ("assistant", "the man was angry again"),
            ],
            torch.tensor([1.0, 1.0, 0.0]),
        ),
        (
            [
                ("user", "what happened to the phone"),
                ("assistant", "the phone was broken"),
            ],
            torch.tensor([0.0, 0.0, 1.0]),
        ),
        (
            [
                ("user", "did the phone work again"),
                ("assistant", "the phone was broken again"),
            ],
            torch.tensor([0.0, 0.0, 1.0]),
        ),
    ]

    samples: List[Sample] = []
    for conversation, label in conversation_specs:
        text = conversation_to_text(conversation)
        samples.append(
            Sample(
                text=text,
                label=label,
                modalities={"text": encode_text(text, vocab_to_idx)},
            )
        )
    return InMemoryDataset(samples)
