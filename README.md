# Intention Jailbreak

Research project for intention analysis and jailbreak studies.

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for Python package management.

### Prerequisites

- uv (already installed)

### Setup

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

Or run Python scripts/notebooks with the project automatically available:

```bash
uv run jupyter notebook
```

## Project Structure

- `configs/` - Configuration files for Hydra
- `notes/` - Markdown notes documenting experimental processes
- `data/` - Data directory (gitignored)

## Dependencies

### Core Dependencies
- transformers - Transformer models and utilities
- torch - PyTorch deep learning framework
- seaborn - Statistical data visualization
- matplotlib - Plotting library
- hydra-core - Configuration management
- pandas - Data manipulation and analysis

### Optional Dependencies
- vllm - Large language model inference engine
