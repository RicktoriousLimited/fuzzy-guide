# NSCTX Demo Application

This repository contains a self-contained demo of the NSCTX (Neuro-Symbolic Contextual Transformer eXperimental) model described in `Main.txt`. The app provides a minimal end-to-end pipeline covering data preparation, training, evaluation, and explanation for a toy multi-modal reasoning task.

## Features

- PyTorch implementation of modality embeddings, role-aware transformers, semantic graph induction, and neuro-symbolic reasoning layers.
- Meta-learning inspired adaptation via Elastic Weight Consolidation (EWC) penalties.
- Command-line interface to train the model on a small in-memory dataset and run predictions with explanations.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Train the demo model and save weights:

```bash
python -m nsctx.cli train --epochs 50 --model-path nsctx_demo.pt
```

Run inference with the trained weights:

```bash
python -m nsctx.cli predict --model-path nsctx_demo.pt --text "the boy apologized"
```

The prediction command prints a three-dimensional probability vector along with optional intermediate representations for inspection.

## Project Structure

- `nsctx/`: Python package containing the model, training utilities, and CLI.
- `Main.txt`: Original mathematical description of NSCTX.
- `requirements.txt`: Python dependencies for the demo.

## Notes

The demo uses a tiny, synthetic dataset and lightweight modules intended for experimentation and educational purposes rather than production use. The architecture mirrors the layered structure described in the specification while remaining computationally light enough to run on CPU in seconds.
