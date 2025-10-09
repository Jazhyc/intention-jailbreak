# Training: Harmful Prompt Classification

## Overview

Binary classifier to distinguish harmful from unharmful prompts using the WildGuardMix dataset.

## Dataset

**Source**: `allenai/wildguardmix` (wildguardtrain subset)  
**Total examples**: 92,250

### Splits
- **Train/Test**: 80/20 (69,407 / 17,352)
- **Train/Val**: 90/10 (62,466 / 6,941)

### Stratification
All splits stratified on:
- `prompt_harm_label` (harmful/unharmful)
- `adversarial` (True/False)
- `subcategory` (benign, violence, etc.)

This ensures consistent distribution across all splits.

## Model

**Architecture**: ModernBERT-base (answerdotai/ModernBERT-base)  
**Parameters**: 150M  
**Task**: Binary sequence classification

## Training Configuration

### Hyperparameters
- Batch size: 256
- Learning rate: 1e-4
- Optimizer: AdamW 8-bit (bitsandbytes)
- Precision: bfloat16
- Epochs: 1
- Max sequence length: 512

### Optimization
- torch.compile enabled for speed
- No gradient accumulation
- No weight decay
- No warmup
- No gradient clipping

### Evaluation
- Validation every 20% of epoch
- Metrics: accuracy, F1, precision, recall
- Best model selection: F1 score

### Logging
- Platform: Weights & Biases
- Entity: intention-analysis
- Project: classifier-training
- Local logs: `logs/wandb/`, `logs/tensorboard/`

## Output

**Model location**: `models/modernbert-classifier/final_model/`  
**Format**: safetensors

## Notes

- Test set reserved for final evaluation (not used during training)
- Configuration managed via Hydra (`configs/train_config.yaml`)
- All settings use YAML anchors to avoid duplication
