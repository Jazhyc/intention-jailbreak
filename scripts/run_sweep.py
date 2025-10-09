"""
Hyperparameter tuning script using Hydra's multirun feature.

This script demonstrates how to run grid search for hyperparameter optimization
using Hydra's built-in multirun capability.

Usage:
    # Run single experiment with default config
    python train.py
    
    # Run grid search with Hydra multirun
    python train.py --multirun \
        training.learning_rate=5e-5,1e-4,2e-4 \
        training.batch_size=16,32,64 \
        training.weight_decay=0.0,0.01,0.1
    
    # Or use the sweep enabled flag
    python train.py sweep.enabled=true
    
Note: Results will be saved in the multirun/ directory with timestamps.
Each run will log to WandB separately for comparison.
"""

if __name__ == "__main__":
    print(__doc__)
