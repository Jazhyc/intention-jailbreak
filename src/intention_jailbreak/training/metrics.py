"""Metrics computation for model evaluation."""

import numpy as np
from typing import Dict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_classification_metrics(eval_pred) -> Dict[str, float]:
    """
    Compute classification metrics for binary classification.
    
    Args:
        eval_pred: Tuple of (predictions, labels)
    
    Returns:
        Dictionary of metric names and values
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='binary')
    precision = precision_score(labels, predictions, average='binary', zero_division=0)
    recall = recall_score(labels, predictions, average='binary', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }
