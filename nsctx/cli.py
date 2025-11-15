"""Command-line interface for the NSCTX demo app."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Sequence, Tuple

import torch

from . import Evaluator, NSCTXModel, Trainer
from .data import build_demo_dataset, conversation_to_text, encode_text
from .model import NSCTXConfig
from .training import TrainingConfig
from .vocab import VOCAB


ConversationArg = Sequence[Tuple[str, str]]


def _parse_conversation(raw: str | None) -> ConversationArg | None:
    if raw is None:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise ValueError("Conversation must be valid JSON.") from exc
    conversation: List[Tuple[str, str]] = []
    for item in payload:
        role = item.get("role")
        content = item.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError("Conversation entries require 'role' and 'content' strings.")
        conversation.append((role, content))
    if not conversation:
        raise ValueError("Conversation must contain at least one message.")
    return conversation


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NSCTX demo app")
    parser.add_argument("command", choices=["train", "predict"], help="Operation to run")
    parser.add_argument("--model-path", default="nsctx_demo.pt", help="Path to save or load the model")
    parser.add_argument(
        "--text",
        default="<user> what happened in the game <assistant> the boy kicked the ball",
        help="Whitespace separated conversation transcript for prediction",
    )
    parser.add_argument(
        "--conversation",
        default=None,
        help="JSON encoded list of {'role': str, 'content': str} messages for prediction",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--device", default="cpu", help="Device to use")
    return parser.parse_args()


def build_model(device: str) -> NSCTXModel:
    config = NSCTXConfig(vocab_size=len(VOCAB))
    model = NSCTXModel(config)
    return model.to(device)


def train(model_path: Path, epochs: int, device: str) -> None:
    dataset = build_demo_dataset(VOCAB)
    model = build_model(device)
    trainer = Trainer(model, TrainingConfig(epochs=epochs, device=device))
    metrics = trainer.fit(dataset)
    torch.save({"state_dict": model.state_dict()}, model_path)
    print("Training complete. Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")


def predict(model_path: Path, text: str, device: str) -> None:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = build_model(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    evaluator = Evaluator(model, device=device)
    vocab_to_idx = {token: i for i, token in enumerate(VOCAB)}
    encoded = encode_text(text, vocab_to_idx)
    tokens = encoded.unsqueeze(0)
    probs = evaluator.predict(tokens)
    print(f"Predictions for '{text}': {probs.squeeze(0).tolist()}")


def main() -> None:
    args = get_args()
    model_path = Path(args.model_path)
    if args.command == "train":
        train(model_path, args.epochs, args.device)
    else:
        conversation = _parse_conversation(args.conversation)
        if conversation is not None:
            encoded = conversation_to_text(conversation)
            text = encoded
        else:
            text = args.text
        predict(model_path, text, args.device)


if __name__ == "__main__":  # pragma: no cover
    main()
