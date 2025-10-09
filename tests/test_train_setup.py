"""
Test script to verify the training setup before running full training.

This script tests:
1. Data loading and preprocessing
2. Model and tokenizer loading
3. Configuration loading
4. Basic forward pass

Usage:
    python tests/test_setup.py
"""

import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from omegaconf import OmegaConf
from intention_jailbreak.dataset import wildguardmix
from intention_jailbreak.training import print_gpu_info


def test_data_loading():
    """Test data loading and splitting."""
    print("\nTesting Data Loading...")
    
    try:
        train_df, test_df = wildguardmix.load_and_split(test_size=0.2)
        print(f"✓ Data loaded: Train {len(train_df)}, Test {len(test_df)}")
        print(f"  Harmful: {(train_df['prompt_harm_label'] == 'harmful').sum()}")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_model_loading():
    """Test model and tokenizer loading."""
    print("\nTesting Model Loading...")
    
    try:
        model_name = "answerdotai/ModernBERT-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            problem_type="single_label_classification",
            dtype=torch.bfloat16
        )
        
        params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"✓ Model loaded: {params:.1f}M parameters")
        return True, tokenizer, model
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False, None, None


def test_forward_pass(tokenizer, model):
    """Test a forward pass through the model."""
    print("\nTesting Forward Pass...")
    
    try:
        test_prompt = "How can I learn to code in Python?"
        inputs = tokenizer(test_prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
        
        prediction = torch.argmax(outputs.logits, dim=-1).item()
        print(f"✓ Forward pass successful: {'harmful' if prediction == 1 else 'unharmful'}")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_mini_training():
    """Test a mini training run to verify the full pipeline."""
    print("\nTesting Mini Training Pipeline...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
        from datasets import Dataset
        from intention_jailbreak.training.metrics import compute_classification_metrics
        import tempfile
        import shutil
        
        # Create tiny synthetic dataset
        train_data = {
            'prompt': ['How do I make a bomb?', 'What is Python?', 'How to hack a computer?', 'Explain gravity'],
            'label': [1, 0, 1, 0]  # 1=harmful, 0=unharmful
        }
        val_data = {
            'prompt': ['How to steal a car?', 'What is machine learning?'],
            'label': [1, 0]
        }
        
        train_dataset = Dataset.from_dict(train_data)
        val_dataset = Dataset.from_dict(val_data)
        
        # Load model and tokenizer
        model_name = "answerdotai/ModernBERT-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            problem_type="single_label_classification",
            torch_dtype=torch.bfloat16
        )
        
        # Tokenize
        def tokenize_function(examples):
            return tokenizer(examples['prompt'], padding='max_length', truncation=True, max_length=128)
        
        train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=['prompt'])
        val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=['prompt'])
        
        # Create temp directory for output
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Mini training config
            import os
            os.environ["WANDB_PROJECT"] = "classifier-test"
            os.environ["WANDB_DIR"] = "logs/wandb"
            
            training_args = TrainingArguments(
                output_dir=temp_dir,
                max_steps=2,
                per_device_train_batch_size=2,
                per_device_eval_batch_size=2,
                learning_rate=1e-4,
                eval_strategy="steps",
                eval_steps=1,
                save_strategy="no",
                logging_dir="logs",
                logging_steps=1,
                report_to="wandb",
                remove_unused_columns=True,
                seed=42,
            )
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                processing_class=tokenizer,
                compute_metrics=compute_classification_metrics,
            )
            
            # Train for 2 steps
            trainer.train()
            
            # Evaluate
            eval_results = trainer.evaluate()
            
            print(f"✓ Mini training successful: F1={eval_results.get('eval_f1', 0):.3f}")
            return True
            
        finally:
            # Clean up temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)
            
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_loading():
    """Test configuration loading."""
    print("\nTesting Configuration Loading...")
    
    try:
        config_path = Path("configs/train_config.yaml")
        cfg = OmegaConf.load(config_path)
        print(f"✓ Config loaded: {cfg.model.name}, batch={cfg.training.per_device_train_batch_size}")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_gpu_availability():
    """Test GPU availability."""
    print("\nTesting GPU...")
    print_gpu_info()
    return torch.cuda.is_available()


def main():
    """Run all tests."""
    print("\nTraining Setup Verification")
    print("-" * 40)
    
    results = []
    results.append(("Data", test_data_loading()))
    results.append(("Config", test_config_loading()))
    results.append(("GPU", test_gpu_availability()))
    
    success, tokenizer, model = test_model_loading()
    results.append(("Model", success))
    
    if success:
        results.append(("Forward Pass", test_forward_pass(tokenizer, model)))
    
    # Run mini training test
    results.append(("Mini Training", test_mini_training()))
    
    print("\nSummary:")
    for name, result in results:
        status = "✓" if result else "✗"
        print(f"  {status} {name}")
    
    all_passed = all(r[1] for r in results if r[0] != "GPU")
    
    if all_passed:
        print("\n✓ All tests passed! Ready to train.")
    else:
        print("\n✗ Some tests failed")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
