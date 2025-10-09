"""
Evaluate trained model on test set and save predictions with probabilities.

This script:
1. Loads the trained model
2. Evaluates on the held-out test set
3. Saves predictions with harmful class probabilities
4. Logs results to wandb (classifier-annotation-set project)

Usage:
    python scripts/evaluate_test.py
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DIR"] = "logs/wandb"

from pathlib import Path
import pandas as pd
import torch
import wandb
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
import numpy as np
from tqdm import tqdm

from intention_jailbreak.dataset import wildguardmix
from intention_jailbreak.training import compute_classification_metrics, print_gpu_info
from intention_jailbreak.common import LABEL_COLUMN, TEXT_COLUMN, POSITIVE_LABEL


def main():
    """Evaluate model on test set."""
    
    # Configuration
    model_path = "models/modernbert-classifier/final_model"
    output_dir = Path("data/test_predictions")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb
    wandb.init(
        project="classifier-annotation-set",
        entity="intention-analysis",
        name="test-set-evaluation",
        dir="logs/wandb",
        config={
            "model_path": model_path,
            "dataset": "wildguardmix",
            "split": "test"
        }
    )
    
    print_gpu_info()
    
    # Load test data
    print("\nLoading test data...")
    _, test_df = wildguardmix.load_and_split(test_size=0.2, random_state=42)
    print(f"Test set size: {len(test_df)}")
    
    # Load model and tokenizer
    print(f"\nLoading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        dtype=torch.bfloat16
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Prepare labels
    test_df['label'] = (test_df[LABEL_COLUMN] == POSITIVE_LABEL).astype(int)
    
    # Get predictions in batches
    print("\nGenerating predictions...")
    batch_size = 96
    all_probs = []
    all_preds = []
    all_labels = test_df['label'].values
    
    with torch.no_grad():
        for i in tqdm(range(0, len(test_df), batch_size), desc="Evaluating"):
            batch_texts = test_df[TEXT_COLUMN].iloc[i:i+batch_size].tolist()
            
            # Tokenize
            inputs = tokenizer(
                batch_texts,
                padding='max_length',
                truncation=True,
                max_length=256,
                return_tensors='pt'
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get predictions
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Convert to probabilities
            probs = torch.softmax(logits, dim=-1)
            harmful_probs = probs[:, 1].float().cpu().numpy()  # Probability of harmful class
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            
            all_probs.extend(harmful_probs)
            all_preds.extend(preds)
    
    # Add predictions to dataframe
    test_df['harmful_probability'] = all_probs
    test_df['predicted_label'] = all_preds
    
    # Compute metrics
    print("\nComputing metrics...")
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    
    metrics = {
        "test/accuracy": accuracy,
        "test/precision": precision,
        "test/recall": recall,
        "test/f1": f1,
        "test/samples": len(test_df)
    }
    
    print("\nTest Set Results:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    
    # Log to wandb
    wandb.log(metrics)
    
    # Log confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_labels, all_preds)
    wandb.log({
        "test/confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=all_labels,
            preds=all_preds,
            class_names=["unharmful", "harmful"]
        )
    })
    
    # Log probability distribution
    wandb.log({
        "test/harmful_probability_dist": wandb.Histogram(all_probs),
        "test/harmful_prob_by_true_label": wandb.plot.histogram(
            wandb.Table(
                data=[[p, l] for p, l in zip(all_probs, all_labels)],
                columns=["harmful_probability", "true_label"]
            ),
            "harmful_probability",
            title="Harmful Probability Distribution by True Label"
        )
    })
    
    # Remove response column if it exists
    if 'response' in test_df.columns:
        test_df = test_df.drop(columns=['response'])
        print("\nRemoved 'response' column from dataset")
    
    # Save predictions
    output_file = output_dir / "test_predictions.parquet"
    test_df.to_parquet(output_file, index=False)
    print(f"\nPredictions saved to: {output_file}")
    
    # Log artifact to wandb
    artifact = wandb.Artifact(
        name="test-predictions",
        type="dataset",
        description="Test set with model predictions and harmful class probabilities"
    )
    artifact.add_file(output_file)
    wandb.log_artifact(artifact)
    
    # Create summary table
    summary_data = []
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        preds_at_threshold = (np.array(all_probs) >= threshold).astype(int)
        acc = accuracy_score(all_labels, preds_at_threshold)
        prec, rec, f1_t, _ = precision_recall_fscore_support(
            all_labels, preds_at_threshold, average='binary', zero_division=0
        )
        summary_data.append([threshold, acc, prec, rec, f1_t])
    
    summary_table = wandb.Table(
        data=summary_data,
        columns=["threshold", "accuracy", "precision", "recall", "f1"]
    )
    wandb.log({"test/metrics_by_threshold": summary_table})
    
    wandb.finish()
    
    print("\nEvaluation complete!")
    print(f"Total samples: {len(test_df)}")
    print(f"Harmful probability column added: 'harmful_probability'")
    print(f"  (Probability that the prompt is classified as harmful)")


if __name__ == "__main__":
    main()
