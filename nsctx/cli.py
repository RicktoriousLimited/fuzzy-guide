"""Command-line interface for the NSCTX demo app."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch

from . import Evaluator, NSCTXModel, Trainer
from .data import build_demo_dataset
from .model import NSCTXConfig
from .training import TrainingConfig


VOCAB: List[str] = [
    "<pad>",
    "<unk>",
    "the",
    "boy",
    "kicked",
    "ball",
    "apologized",
    "man",
    "was",
    "angry",
    "phone",
    "broken",
]


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NSCTX demo app")
    parser.add_argument("command", choices=["train", "predict"], help="Operation to run")
    parser.add_argument("--model-path", default="nsctx_demo.pt", help="Path to save or load the model")
    parser.add_argument("--text", default="the boy apologized", help="Input sentence for prediction")
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
    tokens = torch.tensor([[vocab_to_idx.get(tok, vocab_to_idx["<unk>"]) for tok in text.split()]], dtype=torch.long)
    probs = evaluator.predict(tokens)
    print(f"Predictions for '{text}': {probs.squeeze(0).tolist()}")


def main() -> None:
    args = get_args()
    model_path = Path(args.model_path)
    if args.command == "train":
        train(model_path, args.epochs, args.device)
    else:
        predict(model_path, args.text, args.device)


if __name__ == "__main__":  # pragma: no cover
    main()
