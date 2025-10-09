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
            problem_type="single_label_classification"
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
