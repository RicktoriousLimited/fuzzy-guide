# NSCTX Demo Application

This repository contains a self-contained demo of the NSCTX (Neuro-Symbolic Contextual Transformer eXperimental) model described in `Main.txt`. The app provides a minimal end-to-end pipeline covering data preparation, training, evaluation, and explanation for a toy multi-modal reasoning task.

## Features

- PyTorch implementation of modality embeddings, role-aware transformers, semantic graph induction, and neuro-symbolic reasoning layers.
- Meta-learning inspired adaptation via Elastic Weight Consolidation (EWC) penalties.
- Command-line interfaces in both Python and PHP to train the model on a small in-memory dataset and run predictions with explanations.

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

### PHP NSCTX Playground

The repository now includes a full PHP port of the NSCTX demo. The PHP stack mirrors the mathematical specification and exposes both a browser interface and CLI utilities for training, testing, and reasoning inspection.

Start the development server from the repository root:

```bash
php -S localhost:8080 -t php-app
```

Then visit <http://localhost:8080> to interact with the playground. The UI lets you launch EWC-regularized training on the bundled multi-modal dataset, inspect prototype vectors, and run predictions with reasoning traces.

Run the PHP training pipeline from the command line:

```bash
php php-app/train.php --dataset php-app/data/dataset.json --model php-app/storage/model.json
```

Make a prediction with the trained model:

```bash
php php-app/predict.php --model php-app/storage/model.json \
  --text "the boy apologized to his friend" \
  --image "0.2,0.1,0.7,0.5" \
  --audio "0.3,0.4,0.2"
```

## Testing

A lightweight PHPUnit configuration is included for the PHP components. From the repository root run:

```bash
vendor/bin/phpunit
```

The suite exercises encoders, fusion, reasoning, storage, the conversation-aware model pipeline, CLI flows, and the web entry point.

## Project Structure

- `nsctx/`: Python package containing the model, training utilities, and CLI.
- `php-app/`: Standalone PHP web application implementing the NSCTX playground and CLI tooling.
- `Main.txt`: Original mathematical description of NSCTX.
- `requirements.txt`: Python dependencies for the demo.

## Notes

The demo uses a tiny, synthetic dataset and lightweight modules intended for experimentation and educational purposes rather than production use. The architecture mirrors the layered structure described in the specification while remaining computationally light enough to run on CPU in seconds.
