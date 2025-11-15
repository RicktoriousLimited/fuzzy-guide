"""Conversation-friendly transformer chatbot built on NSCTX."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from .data import ConversationMessage, InMemoryDataset, build_demo_dataset, conversation_to_text, encode_text
from .evaluation import Evaluator
from .model import NSCTXConfig, NSCTXModel
from .training import Trainer, TrainingConfig
from .vocab import VOCAB


@dataclass
class MemoryEntry:
    """Container for a learned conversation snippet."""

    messages: List[ConversationMessage]
    text: str
    embedding: torch.Tensor


class ConversationChatbot:
    """High-level helper that turns the NSCTX model into a chatbot."""

    def __init__(self, vocab: Iterable[str] | None = None, device: str = "cpu"):
        self.vocab = list(vocab or VOCAB)
        if "<unk>" not in self.vocab:
            raise ValueError("Vocabulary must contain a <unk> token for unseen words.")
        self.vocab_to_idx = {token: i for i, token in enumerate(self.vocab)}
        config = NSCTXConfig(vocab_size=len(self.vocab))
        self.device = device
        self.model = NSCTXModel(config).to(device)
        self.evaluator = Evaluator(self.model, device=device)
        self.memory: List[MemoryEntry] = []

    def train_on_dataset(self, dataset: InMemoryDataset | None = None, epochs: int = 20) -> Dict[str, float]:
        """Train (or fine-tune) the underlying transformer on a dataset."""

        dataset = dataset or build_demo_dataset(self.vocab)
        trainer = Trainer(self.model, TrainingConfig(epochs=epochs, device=self.device))
        return trainer.fit(dataset)

    # ------------------------------------------------------------------
    # Memory handling
    # ------------------------------------------------------------------
    def learn(self, conversation: Sequence[ConversationMessage] | str) -> Dict[str, object]:
        """Store a conversation in memory for later retrieval."""

        normalized = self._normalise_conversation(conversation)
        text = conversation_to_text(normalized)
        embedding = self._embed_text(text)
        entry = MemoryEntry(messages=normalized, text=text, embedding=embedding)
        self.memory.append(entry)
        return {
            "text": text,
            "messages": normalized,
        }

    # ------------------------------------------------------------------
    # Chatting
    # ------------------------------------------------------------------
    def respond(self, conversation: Sequence[ConversationMessage] | str) -> Dict[str, object]:
        """Generate a response and expose reasoning artefacts."""

        normalized = self._normalise_conversation(conversation)
        text = conversation_to_text(normalized)
        tokens = encode_text(text, self.vocab_to_idx).unsqueeze(0)
        probs = self.evaluator.predict(tokens).squeeze(0)
        embedding = self._embed_text(text)
        memory, score = self._retrieve_memory(embedding)
        response = self._compose_response(normalized, memory, score, probs)
        return {
            "response": response,
            "probabilities": probs.tolist(),
            "memory": self._serialise_memory(memory),
            "match_score": score,
            "summary": text,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _normalise_conversation(self, conversation: Sequence[ConversationMessage] | str) -> List[ConversationMessage]:
        if isinstance(conversation, str):
            return [("user", conversation.strip())]
        normalised: List[ConversationMessage] = []
        for role, content in conversation:
            role_clean = role.strip().lower() or "user"
            content_clean = content.strip()
            if content_clean:
                normalised.append((role_clean, content_clean))
        if not normalised:
            raise ValueError("Conversation must contain at least one non-empty turn.")
        return normalised

    def _embed_text(self, text: str) -> torch.Tensor:
        tokens = encode_text(text, self.vocab_to_idx).unsqueeze(0).to(self.device)
        mask = (tokens != 0).float()
        with torch.no_grad():
            projected, _ = self.model.encode_text(tokens, mask)
            normalized = F.normalize(projected.squeeze(0), dim=0)
            return normalized.cpu()

    def _retrieve_memory(self, embedding: torch.Tensor) -> Tuple[MemoryEntry | None, float]:
        best_entry: MemoryEntry | None = None
        best_score = 0.0
        if not self.memory:
            return None, 0.0
        for entry in self.memory:
            score = F.cosine_similarity(entry.embedding.unsqueeze(0), embedding.unsqueeze(0)).item()
            if best_entry is None or score > best_score:
                best_entry = entry
                best_score = score
        if best_entry is None or best_score < 0.05:
            return None, 0.0
        return best_entry, best_score

    def _compose_response(
        self,
        conversation: Sequence[ConversationMessage],
        memory: MemoryEntry | None,
        score: float,
        probs: torch.Tensor,
    ) -> str:
        intent_line = self._describe_user_intent(conversation)
        reasoning_line = self._describe_probabilities(probs)
        if memory is not None:
            memory_line = self._describe_memory_reference(memory, score)
            return " ".join(part for part in [intent_line, memory_line, reasoning_line] if part).strip()
        exploration_line = (
            "I do not have a closely matching archive entry yet, so I am reasoning directly from the transformer context."
        )
        return " ".join(part for part in [intent_line, exploration_line, reasoning_line] if part).strip()

    def _describe_user_intent(self, conversation: Sequence[ConversationMessage]) -> str:
        for role, content in reversed(conversation):
            if role == "user" and content:
                return f"You are focusing on: '{content}'."
        # Fallback to the final turn if no explicit user turn was found (e.g. user text only)
        last_turn = conversation[-1]
        return f"Let's continue thinking about: '{last_turn[1]}'."

    def _describe_memory_reference(self, memory: MemoryEntry, score: float) -> str:
        preview = memory.text.strip()
        if len(preview) > 120:
            preview = preview[:117].rstrip() + "..."
        return f"I recall a related memory (match {score:.2f}): '{preview}'."

    def _describe_probabilities(self, probs: torch.Tensor) -> str:
        values = probs.tolist()
        formatted_probs = ", ".join(f"{value:.2f}" for value in values)
        if not values:
            return ""
        top_idx = int(torch.tensor(values).argmax().item())
        top_conf = values[top_idx]
        return (
            "My current reasoning leans toward hypothesis "
            f"{top_idx + 1} with confidence {top_conf:.2f} (distribution [{formatted_probs}])."
        )

    def _serialise_memory(self, memory: MemoryEntry | None) -> Dict[str, object] | None:
        if memory is None:
            return None
        return {
            "text": memory.text,
            "messages": memory.messages,
        }


__all__ = ["ConversationChatbot"]
