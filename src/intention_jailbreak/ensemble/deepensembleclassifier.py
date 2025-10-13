import json
from pathlib import Path
from typing import Callable
import torch
from torch import nn



class DeepEnsembleClassifier(nn.Module):
    """
    An implementation of an ensemble for providing a rough estimate of the Bayesian predictive posterior
    distribution p(y|x, D).
    It is a simple wrapper class that instantiates N models
    """
    def __init__(self, model_fn: Callable, num_models: int):
        super().__init__()
        self.models = nn.ModuleList([model_fn() for _ in range(num_models)])
        self.num_models = num_models
    
    def forward(self, **kwargs):
        """Forward pass through all models and return averaged output."""
        # Remove labels and weight if present (they shouldn't be in forward)
        labels = kwargs.pop('labels', None)
        weight = kwargs.pop('weight', None)
        
        # Get outputs from all models
        outputs_list = [model(**kwargs) for model in self.models]
        
        # Stack logits and average
        logits = torch.stack([out.logits for out in outputs_list])  # Shape: (M, batch, num_classes)
        mean_logits = logits.mean(dim=0)
        
        # Return in the same format as a single model
        # Use the first model's output as a template
        result = type(outputs_list[0])(logits=mean_logits)
        return result

    def save_pretrained(self, save_directory, safe_serialization=True):
        """Save each model in the ensemble to separate directories."""
        Path(save_directory).mkdir(parents=True, exist_ok=True)
        
        for i, model in enumerate(self.models):
            model_dir = Path(save_directory) / f"model_{i}"
            model.save_pretrained(model_dir, safe_serialization=safe_serialization)
        
        # Save ensemble configuration
        config = {
            "num_models": self.num_models,
            "ensemble_type": "DeepEnsembleClassifier"
        }
        with open(Path(save_directory) / "ensemble_config.json", "w") as f:
            json.dump(config, f)
    
    @classmethod
    def from_pretrained(cls, save_directory, model_fn):
        """Load ensemble from saved models."""
        save_path = Path(save_directory)
        
        # Load configuration
        with open(save_path / "ensemble_config.json", "r") as f:
            config = json.load(f)
        
        num_models = config["num_models"]
        ensemble = cls(model_fn, num_models)
        
        # Load each model
        for i in range(num_models):
            model_dir = save_path / f"model_{i}"
            ensemble.models[i] = ensemble.models[i].from_pretrained(model_dir)
        
        return ensemble
