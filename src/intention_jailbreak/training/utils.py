"""Utility functions for setting seeds and printing system info."""

import numpy as np
import torch
from typing import Dict, Optional
from transformers import set_seed, Trainer

from intention_jailbreak.ensemble.deepensembleclassifier import DeepEnsembleClassifier

torch.set_float32_matmul_precision('high')

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


class WeightedTrainer(Trainer):
    """
    Custom Trainer that applies per-sample weights to the loss function.
    
    Expects the dataset to have a 'weight' column with per-sample weights.
    
    Args:
        *args, **kwargs: Passed to parent Trainer
    """
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute weighted cross-entropy loss using per-sample weights from dataset.
        """
        labels = inputs.pop("labels")
        # Extract weights if present in inputs
        weights = inputs.pop("weight", None)
        
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Compute per-sample loss (reduction='none')
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        per_sample_loss = loss_fct(logits, labels)
        
        # Apply sample weights if provided
        if weights is not None:
            weights = weights.to(per_sample_loss.device)
            weighted_loss = per_sample_loss * weights
            loss = weighted_loss.mean()
        else:
            loss = per_sample_loss.mean()
        
        return (loss, outputs) if return_outputs else loss

class SequentialEnsembleTrainer(Trainer):
    def __init__(self, model: DeepEnsembleClassifier, *args, **kwargs):
        super().__init__(model=model, *args, **kwargs)
        self.ensemble = model
        self.current_model_idx = 0
        self.models_trained = [False] * len(self.ensemble.models)
    
    def training_step(self, model, inputs):
        """Only train one model at a time per training step"""
        # Get the current model to train
        single_model = self.ensemble.models[self.current_model_idx]
        
        # Perform standard training step on just this model
        loss = self.compute_loss(single_model, inputs, return_outputs=False)
        
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        
        loss.backward()
        
        return loss.detach()
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Rotate to next model after each epoch"""
        # Mark current model as trained
        self.models_trained[self.current_model_idx] = True
        
        # Move to next model
        self.current_model_idx = (self.current_model_idx + 1) % len(self.ensemble.models)
        
        print(f"Epoch {state.epoch} complete. Now training model {self.current_model_idx}")
        
        super().on_epoch_end(args, state, control, **kwargs)
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute loss for single model (same as the WeightedTrainer)"""
        labels = inputs.pop("labels")
        weights = inputs.pop("weight", None)
        
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Compute per-sample loss
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        per_sample_loss = loss_fct(logits, labels)
        
        # Apply sample weights if provided
        if weights is not None:
            weights = weights.to(per_sample_loss.device)
            weighted_loss = per_sample_loss * weights
            loss = weighted_loss.mean()
        else:
            loss = per_sample_loss.mean()
        
        return (loss, outputs) if return_outputs else loss
