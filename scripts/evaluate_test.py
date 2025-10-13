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

from intention_jailbreak.ensemble.deepensembleclassifier import DeepEnsembleClassifier
from intention_jailbreak.utils.calibration import classifier_calibration_curve, classifier_calibration_error, plot_calibration_curve
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
from intention_jailbreak.training import compute_classification_metrics, print_gpu_info, tokenize_dataset
from intention_jailbreak.common import LABEL_COLUMN, TEXT_COLUMN, POSITIVE_LABEL


# Configuration Constants
MODEL_PATH = "models/modernbert-ensemble/final_model"
USE_ENSEMBLE = True
NUM_ENSEMBLE_MODELS = 3
BATCH_SIZE = 512
MAX_LENGTH = 2048
TEST_SIZE = 0.8  # 80% test split
RANDOM_STATE = 42
FILTER_ENGLISH = True
LANGUAGE_CACHE_DIR = "data/cache"
OUTPUT_DIR = Path("data/test_predictions")
WANDB_PROJECT = "classifier-annotation-set"
WANDB_ENTITY = "intention-analysis"
WANDB_RUN_NAME = "test-set-evaluation-ensemble"
NUM_CALIBRATION_BINS = 20
EVAL_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]
NUM_WORKERS = 8  # Number of workers for data loading
PIN_MEMORY = True  # Pin memory for faster GPU transfer


def main():
    """Evaluate model on test set."""
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=WANDB_RUN_NAME,
        dir="logs/wandb",
        config={
            "model_path": MODEL_PATH,
            "use_ensemble": USE_ENSEMBLE,
            "num_ensemble_models": NUM_ENSEMBLE_MODELS if USE_ENSEMBLE else 1,
            "batch_size": BATCH_SIZE,
            "max_length": MAX_LENGTH,
            "test_size": TEST_SIZE,
            "filter_english": FILTER_ENGLISH,
            "num_workers": NUM_WORKERS,
            "pin_memory": PIN_MEMORY,
            "dataset": "wildguardmix",
            "split": "test"
        }
    )
    
    print_gpu_info()
    
    # Load test data
    print("\nLoading test data...")
    _, test_df = wildguardmix.load_and_split(
        test_size=TEST_SIZE,
        random_state=42,
        filter_english=FILTER_ENGLISH,
        text_column='prompt',
        language_cache_dir=LANGUAGE_CACHE_DIR
    )
    print(f"Test set size: {len(test_df)}")
    
    # Load model and tokenizer
    print(f"\nLoading model from: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    if USE_ENSEMBLE:
        model = DeepEnsembleClassifier.from_pretrained(
            MODEL_PATH,
            model_class=AutoModelForSequenceClassification,
            dtype=torch.bfloat16
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_PATH,
            dtype=torch.bfloat16
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model = torch.compile(model)
    model.eval()
    
    # Prepare labels
    test_df['label'] = (test_df[LABEL_COLUMN] == POSITIVE_LABEL).astype(int)
    
    # Create dataset
    print("\nPreparing dataset...")
    test_dataset = Dataset.from_pandas(test_df[[TEXT_COLUMN, 'label']])
    
    # Tokenize using the same function as training script
    print("Tokenizing...")
    test_dataset = tokenize_dataset(
        test_dataset,
        tokenizer,
        MAX_LENGTH,
        TEXT_COLUMN,
        num_proc=NUM_WORKERS
    )
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    
    # Create DataLoader with multiple workers
    from torch.utils.data import DataLoader
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False
    )
    
    # Get predictions in batches
    print("\nGenerating predictions...")
    all_probs = []
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label']
            
            # Get predictions
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Handle ensemble vs single model outputs
            if hasattr(outputs, 'probs'):  # Ensemble model
                probs = outputs.probs
            else:  # Single model
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
            
            harmful_probs = probs[:, 1].float().cpu().numpy()  # Probability of harmful class
            preds = torch.argmax(probs, dim=-1).cpu().numpy()
            
            all_probs.extend(harmful_probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
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
    output_file = OUTPUT_DIR / "test_predictions.parquet"
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

    # Compute classifier calibration curve
    conf_curve, acc_curve = classifier_calibration_curve(
        y_pred=np.array(all_preds),
        y_true=np.array(all_labels),
        y_confidences=np.array(all_probs),
        num_bins=NUM_CALIBRATION_BINS
    )

    # Compute calibration error
    cal_error = classifier_calibration_error(
        y_pred=np.array(all_preds),
        y_true=np.array(all_labels),
        y_confidences=np.array(all_probs),
        num_bins=NUM_CALIBRATION_BINS,
    )

    cal_curve_path = OUTPUT_DIR / "classifier_calibration_curve.png"

    plot_calibration_curve(
        conf=conf_curve,
        acc=acc_curve,
        title="Classifier Calibration Curve",
        y_label="Empirical Accuracy",
        relative_save_path=cal_curve_path
    )

    print(f"Classifier Calibration Error: {cal_error:.4f}")
    wandb.log({
        "test/calibration_error_weighted": cal_error
    })

    artifact_cal = wandb.Artifact(
        name="classifier_calibration_curve",
        type="figure",
        description="Classifier calibration curve (probabilities vs empirical accuracy)"
    )
    artifact_cal.add_file(cal_curve_path)
    wandb.log_artifact(artifact_cal)
    
    wandb.finish()
    
    print("\nEvaluation complete!")
    print(f"Total samples: {len(test_df)}")
    print(f"Harmful probability column added: 'harmful_probability'")
    print(f"  (Probability that the prompt is classified as harmful)")


if __name__ == "__main__":
    main()
