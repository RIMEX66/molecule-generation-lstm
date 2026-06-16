# Molecule Generation Challenge

Generation of novel molecular SMILES strings using a token-level LSTM language model implemented in PyTorch.

## Overview

This project was developed as part of the *Artificial Intelligence in Life Sciences* course. The objective was to generate realistic and novel molecular structures represented as SMILES strings while minimizing the Fréchet ChemNet Distance (FCD).

The workflow consists of:

1. Dataset analysis
2. SMILES tokenization
3. Vocabulary construction
4. LSTM training
5. Molecule generation
6. Filtering for validity, uniqueness, and novelty
7. Submission creation and evaluation

## Requirements

- Python 3.11
- PyTorch
- RDKit
- NumPy

## Dataset

The model was trained on a dataset containing **1,272,851** molecular SMILES strings.

Dataset statistics:

* Average SMILES length: 47.57
* Median length: 46
* Maximum length: 100
* Vocabulary size: 99 tokens

A maximum sequence length of 120 was used during training to accommodate special tokens and provide a safety margin.

## Model Architecture

The generator is implemented as a token-level LSTM language model.

Architecture:

```text
Input Tokens
    ↓
Embedding Layer (99 → 256)
    ↓
2-Layer LSTM (hidden size 512)
    ↓
Linear Output Layer (512 → 99)
    ↓
Next-Token Prediction
```

The model is trained autoregressively to predict the next token in a SMILES sequence.

## Training Configuration

* Embedding dimension: 256
* Hidden dimension: 512
* Number of LSTM layers: 2
* Dropout: 0.2
* Batch size: 512
* Epochs: 15
* Maximum sequence length: 120
* Optimizer: Adam
* Learning rate: 0.001

Training was performed on the full dataset using an NVIDIA RTX 4070 Ti SUPER GPU.

## Technologies

* Python
* PyTorch
* RDKit
* NumPy

## Skills Demonstrated

- Deep Learning
- Sequence Modeling
- Generative AI
- PyTorch
- Data Preprocessing
- RDKit
- Model Evaluation

## Results

Official challenge evaluation:

* FCD: 0.521
* Validity: 1.000
* Uniqueness: 1.000
* Novelty: 1.000

The final submission was evaluated using the official challenge evaluation pipeline.

The model successfully generated valid, unique, and novel molecular structures while achieving a low Fréchet ChemNet Distance (FCD) score.

## Repository Structure

```text
src/
├── analyze_dataset.py
├── smiles_tokenizer.py
├── build_vocab.py
├── dataset.py
├── model.py
├── train_lstm.py
├── generate_smiles.py
├── create_submission.py
└── check_metrics.py

models/
└── vocab.json
```

## How to Run

### 1. Build the Vocabulary

```bash
python src/build_vocab.py
```

This creates:

```text
models/vocab.json
```

### 2. Train the Model

```bash
python src/train_lstm.py
```

This trains the LSTM model and saves checkpoints in:

```text
models/
```

### 3. Generate Molecules

```bash
python src/generate_smiles.py
```

This generates molecular SMILES strings and saves them to:

```text
submissions/generated_raw.txt
```

### 4. Create the Final Submission

```bash
python src/create_submission.py
```

This filters generated molecules for:

* Validity
* Uniqueness
* Novelty

and creates:

```text
submissions/submission_final.txt
```

### 5. Evaluate the Submission

```bash
python src/check_metrics.py
```

This computes:

* Validity
* Uniqueness
* Novelty

using RDKit and the training dataset.

## Notes

The training dataset and evaluation package are not included in this repository.

The trained checkpoint used for the final submission is not included due to file size limitations.
