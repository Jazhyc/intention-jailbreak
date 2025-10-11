# Intention Jailbreak

Research project for intention analysis and jailbreak studies.

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for Python package management.

1. Install Python 3.12, create a virtual environment, and install the project in editable mode with all dependencies:

```bash
uv sync
```

This will install the `intention_jailbreak` package in editable mode, so changes to the source code are immediately reflected.

2. (Optional) Install the vllm dependency:

```bash
uv sync --extra vllm
```

3. Activate the virtual environment:

```bash
source .venv/bin/activate
```

Alternatively, you can run commands directly with uv without activating:

```bash
uv run python your_script.py
```

For faster training, install flash attention

## Project Structure

```
intention-jailbreak/
├── configs/                      # Hydra configuration files
│   ├── train_config.yaml        # Training configuration
│   └── sweep_config.yaml        # Hyperparameter sweep config
├── data/                         # Data directory (gitignored)
├── logs/                         # Training logs (gitignored)
├── models/                       # Trained models (gitignored)
├── notebooks/                    # Jupyter notebooks for analysis
├── notes/                        # Markdown documentation
├── scripts/                      # Training scripts
│   ├── train.py                 # Main training script
│   └── run_sweep.py             # Sweep helper
├── src/intention_jailbreak/      # Main Python library
│   ├── dataset/                 # Dataset utilities
│   └── training/                # Training utilities
├── tests/                        # Test scripts
└── pyproject.toml               # Project dependencies
```

## Quick Start

### Data Analysis
```bash
jupyter notebook notebooks/wildguardmix_analysis.ipynb
```

### Training
```bash
# Test setup first
python tests/test_train_setup.py

# Train with default config
python scripts/train.py

# Override config
python scripts/train.py training.per_device_train_batch_size=128

# Hyperparameter sweep
python scripts/train.py --multirun training.learning_rate=5e-5,1e-4,2e-4
```

See `scripts/README.md` and `tests/README.md` for more details.
