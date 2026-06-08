# PokemonMoveGeneration

A Pokémon move text generation project built using a T5-base model, Transformers, and Gradio.

## Overview

This repository fine-tunes a T5 model to generate Pokémon move metadata from natural language descriptions. It includes:

- `app.py`: a Gradio web app for interactive Pokémon move prediction.
- `train.py`: training script that fine-tunes `t5-base` on move descriptions.
- `evaluate.py`: evaluation script that computes BLEU score and prints sample predictions.
- `dataset/dataloader.py`: data preprocessing and PyTorch dataloader creation.
- `data/`: source CSV files containing Pokémon moves and augmented descriptions.

## Features

- Predicts move name, type, power, and accuracy from text descriptions
- Uses a custom input/target format for T5:
  - Input: `predict move: <description>`
  - Target: `move: <name> | type: <type> | power: <power> | accuracy: <accuracy>`
- Provides a Gradio interface for easy text-based interaction

## Repository Structure

- `app.py` — launch the Gradio frontend
- `train.py` — train the T5 model and save weights
- `evaluate.py` — evaluate the trained model on the test split
- `dataset/dataloader.py` — load, preprocess, tokenize, and batch data
- `data/metadata_pokemon_moves.csv` — original move metadata
- `data/augmented_moves.csv` — augmented move descriptions

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

Required packages include:

- `torch`
- `transformers`
- `pandas`
- `scikit-learn`
- `gradio`
- `nltk` (for evaluation BLEU scoring)

## Setup

1. Ensure the `data/` folder contains the CSV datasets.
2. Install dependencies.
3. If you want to train from scratch, run `train.py`.

## Usage

### Run the Gradio app

```bash
python app.py
```

Open the displayed URL in your browser and enter a Pokémon move description to see the generated move, type, power, and accuracy.

### Train a new model

```bash
python train.py
```

The best model is saved automatically to `saved_model/` when validation loss improves.

### Evaluate the model

```bash
python evaluate.py
```

This script computes a BLEU score on the test split and prints sample generated outputs.

## Notes

- The code is designed to use GPU when available, but it can also run on CPU.
- Training uses a T5-base model with early stopping based on validation loss.
- The current data loader turns each description into a formatted input and target pair.
