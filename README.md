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

## Project Structure

- `configs/` - Configuration files for Hydra
- `notes/` - Markdown notes documenting experimental processes
- `data/` - Data directory (gitignored)
