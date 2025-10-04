# Intention Jailbreak

Research project for intention analysis and jailbreak studies.

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for Python package management.

### Prerequisites

- uv (already installed)

### Setup

1. Install Python 3.12 and create a virtual environment with all dependencies:

```bash
uv sync
```

2. Install the optional vllm dependency:

```bash
uv sync --extra vllm
```

3. Activate the virtual environment:

```bash
source .venv/bin/activate
```

Alternatively, you can run commands directly with uv:

```bash
uv run python your_script.py
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
