# Answer Normalizer

RoBERTa-based answer normalization for the Warehouse Spatial Intelligence challenge.

## Overview

This module normalizes freeform answers into their canonical form based on question category:

- **left_right**: Extract spatial relationship (left, right, front, behind, above, below)
- **count**: Extract count as number
- **distance**: Extract distance value in meters
- **mcq**: Extract region index

## Architecture

- **Model**: RoBERTa-base with token classification head
- **Task**: BIO tagging to identify answer spans in freeform text
- **Input**: `[CLS] freeform_answer [SEP] category [SEP]`
- **Output**: Token labels (B=Begin, I=Inside, O=Outside)

## Installation

```bash
pip install -r answer_normalizer/requirements.txt
```

## Data Preparation

Extract answer pairs from dataset:

```bash
# Extract from training data
python answer_normalizer/utils/extract_data.py \
    --input data/train_sample/train_sample.json \
    --output data/train_answer_pairs.json

# Extract from validation data
python answer_normalizer/utils/extract_data.py \
    --input data/val.json \
    --output data/val_answer_pairs.json
```

## Training

Train the model:

```bash
python answer_normalizer/train.py \
    --train_path data/train_answer_pairs.json \
    --val_path data/val_answer_pairs.json \
    --output_dir checkpoints \
    --batch_size 16 \
    --epochs 5 \
    --lr 2e-5
```

### Training Arguments

- `--train_path`: Path to training data JSON
- `--val_path`: Path to validation data JSON
- `--model_name`: HuggingFace model (default: roberta-base)
- `--output_dir`: Output directory for checkpoints
- `--batch_size`: Batch size (default: 16)
- `--epochs`: Number of epochs (default: 5)
- `--lr`: Learning rate (default: 2e-5)
- `--max_length`: Max sequence length (default: 256)

## Usage

### Command Line

```bash
python answer_normalizer/predict.py \
    --answer "The pallet [Region 0] is to the left of the pallet [Region 1]." \
    --category left_right \
    --model_path checkpoints
```

### Python API

```python
from answer_normalizer import ml_based_normalize_answer

# Normalize an answer
result = ml_based_normalize_answer(
    freeform_answer="The distance is 8.66 meters.",
    category="distance"
)
print(result)  # Output: 8.66
```

### Using the function-based API

```python
from answer_normalizer import ml_based_normalize_answer

# Normalize answers using ML-based approach
examples = [
    ("The pallet [Region 0] is to the left.", "left_right"),
    ("There are 4 pallets in the buffer.", "count"),
    ("The distance is 13.65 meters.", "distance"),
    ("Region [3] is the closest.", "mcq")
]

for answer, category in examples:
    result = ml_based_normalize_answer(
        freeform_answer=answer,
        category=category,
        model_path='checkpoints'
    )
    print(f"{category}: {answer} -> {result}")
```

## Examples

### Input/Output Pairs

| Category | Freeform Answer | Normalized Answer |
|----------|----------------|-------------------|
| left_right | "The pallet [Region 0] is to the left of the pallet [Region 1]." | left |
| count | "There are four pallets sitting in Buffer Zone 1." | 4 |
| distance | "The pallet [Region 0] is 8.66 meters away from the pallet [Region 1]." | 8.66 |
| mcq | "The pallet [Region 5] is the closest to transporter [Region 0]." | 5 |

## Normalization Strategy

The model uses a hybrid approach:

1. **Primary**: RoBERTa token classification to identify answer spans
2. **Fallback**: Regex-based extraction when model has low confidence
3. **Post-processing**: Category-specific cleanup (e.g., remove units, normalize case)

### Category-Specific Rules

- **left_right**: Lowercased spatial relation keyword
- **count**: Numeric value (converts word numbers to digits)
- **distance**: Numeric value only (removes "meters" and other units)
- **mcq**: Region index as integer

## Model Performance

The model achieves token-level accuracy on answer span identification. Evaluation metrics are saved in `checkpoints/history.json` after training.

## Files Structure

```
answer_normalizer/
├── __init__.py              # Package initialization
├── model.py                 # RoBERTa model definition
├── dataset.py               # Dataset and dataloader
├── train.py                 # Training script
├── predict.py               # Inference script
├── requirements.txt         # Dependencies
├── README.md               # Documentation
├── utils/
│   └── extract_data.py     # Data extraction utility
└── checkpoints/            # Model checkpoints (created during training)
    ├── best_model.pt       # Best model weights
    ├── tokenizer/          # Tokenizer files
    └── history.json        # Training history
```

## Notes

- The model is trained on a small dataset (100 train, 630 val samples)
- For better performance, train on the full training set (217k samples)
- The fallback regex extraction ensures robustness even with limited training data
- Token classification approach allows extracting answers not seen during training
