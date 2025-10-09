"""
Training script for WildGuardMix binary classifier using ModernBERT.

Usage:
    python scripts/train.py
    python scripts/train.py training.per_device_train_batch_size=128
    python scripts/train.py --multirun training.learning_rate=5e-5,1e-4,2e-4
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import wandb
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

from intention_jailbreak.dataset import wildguardmix
from intention_jailbreak.training import (
    set_all_seeds,
    prepare_classification_data,
    tokenize_dataset,
    compute_classification_metrics,
    print_gpu_info,
)


@hydra.main(version_base=None, config_path="../configs", config_name="train_config")
def main(cfg: DictConfig):
    """Main training function."""
    
    set_all_seeds(cfg.seed)
    print_gpu_info()
    
    # Initialize WandB
    if cfg.training.report_to == "wandb":
        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            dir=cfg.wandb.dir,
            name=cfg.wandb.run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=cfg.wandb.tags,
        )
    
    # Load data
    print("Loading dataset...")
    train_df, test_df = wildguardmix.load_and_split(
        subset=cfg.dataset.subset,
        test_size=cfg.dataset.test_size,
        random_state=cfg.dataset.random_state
    )
    
    # Prepare data (no test set used during training)
    print("Preparing data...")
    train_dataset, val_dataset = prepare_classification_data(
        train_df=train_df,
        test_df=test_df,
        val_size=cfg.dataset.val_size,
        label_column=cfg.dataset.label_column,
        text_column=cfg.dataset.text_column,
        positive_label=cfg.dataset.positive_label,
        random_state=cfg.dataset.random_state
    )
    
    # Load model
    print(f"Loading model: {cfg.model.name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.name,
        dtype=torch.bfloat16,
        **{k: v for k, v in cfg.model.items() if k != 'name' and k != 'max_length'}
    )
    
    # Tokenize
    print("Tokenizing...")
    train_dataset = tokenize_dataset(train_dataset, tokenizer, cfg.model.max_length, cfg.dataset.text_column, num_proc=cfg.dataset.num_proc)
    val_dataset = tokenize_dataset(val_dataset, tokenizer, cfg.model.max_length, cfg.dataset.text_column, num_proc=cfg.dataset.num_proc)
    
    # Setup training
    training_args = TrainingArguments(**cfg.training)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_classification_metrics,
    )
    
    # Train
    print("Training...")
    trainer.train()
    
    # Final model results
    trainer.evaluate()
    
    # Save model
    print("Saving model...")
    final_model_dir = Path(cfg.training.output_dir) / "final_model"
    model.save_pretrained(final_model_dir, safe_serialization=True)
    tokenizer.save_pretrained(final_model_dir)
    print(f"Model saved to: {final_model_dir}")
    
    if cfg.training.report_to == "wandb":
        wandb.finish()
    
    print("Training complete!")


if __name__ == "__main__":
    main()
