"""Utility functions for setting seeds and printing system info."""

import numpy as np
import torch
from transformers import set_seed


def set_all_seeds(seed: int):
    """
    Set all random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    set_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_gpu_info():
    """Print GPU information if available."""
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"CUDA: {torch.version.cuda}")
    else:
        print("⚠️  No GPU available, training will use CPU")
