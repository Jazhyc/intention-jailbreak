# Scripts

Training scripts for the WildGuardMix classifier project.

## Available Scripts

### `train.py`
Main training script for the binary classifier.

**Usage:**
```bash
# Run from project root
python scripts/train.py

# Override config values
python scripts/train.py training.per_device_train_batch_size=128

# Grid search
python scripts/train.py --multirun training.learning_rate=5e-5,1e-4,2e-4
```

**Features:**
- Trains ModernBERT-base for harmful/unharmful classification
- Uses 8-bit Adam optimizer (bitsandbytes)
- Supports torch.compile for speed
- Logs to WandB and TensorBoard
- Saves models in safetensors format

### `run_sweep.py`
Documentation for running hyperparameter sweeps.

## Configuration

Scripts use `configs/train_config.yaml` for all settings. The config uses YAML anchors to avoid duplication (e.g., seed is defined once and referenced elsewhere).

## Output Directories

All outputs are in the `logs/` directory:
- `logs/wandb/`: WandB logs
- `logs/hydra/`: Hydra run outputs
- `models/`: Trained models (outside logs)
