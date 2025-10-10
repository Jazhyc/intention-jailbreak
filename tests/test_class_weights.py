"""Test class weighting functionality."""

import numpy as np
import pandas as pd
from intention_jailbreak.training import compute_sample_weights


def test_compute_class_weights():
    """Test that sample weights are computed correctly."""
    # Create a simple test dataframe with imbalanced subcategories and labels
    # label=0: 4 samples (40%), label=1: 6 samples (60%)
    # benign: 4 samples, violence: 3 samples, sexual: 3 samples
    df = pd.DataFrame({
        'label': [0, 0, 0, 1, 1, 1, 0, 1, 1, 1],
        'subcategory': ['benign', 'benign', 'benign', 'violence', 'violence', 
                       'sexual', 'benign', 'violence', 'sexual', 'sexual']
    })
    
    # Test with both label and subcategory weights
    weights = compute_sample_weights(
        df, 
        label_column='label', 
        weight_column='subcategory',
        use_label_weights=True,
        use_subcategory_weights=True
    )
    
    # Check that we have one weight per sample
    assert len(weights) == len(df)
    
    # Check that all weights are positive
    assert all(w > 0 for w in weights)
    
    # Mean should be normalized to 1.0
    assert abs(weights.mean() - 1.0) < 0.01
    
    # Sample from minority class (label=0) and rare subcategory should have higher weight
    # First sample: label=0 (minority), benign (most common subcat)
    # 6th sample: label=1 (majority), sexual (less common subcat)
    benign_minority_weight = weights[0]  # label=0, benign
    sexual_majority_weight = weights[5]  # label=1, sexual
    
    print(f"✓ Sample weights computed: min={weights.min():.3f}, max={weights.max():.3f}, mean={weights.mean():.3f}")
    print(f"  Sample 0 (label=0, benign): {benign_minority_weight:.3f}")
    print(f"  Sample 5 (label=1, sexual): {sexual_majority_weight:.3f}")


def test_class_weights_balanced():
    """Test sample weights on balanced dataset."""
    # Perfectly balanced dataset (equal counts per label and subcategory)
    df = pd.DataFrame({
        'label': [0, 0, 1, 1, 0, 0, 1, 1],
        'subcategory': ['benign', 'benign', 'violence', 'violence', 
                       'benign', 'benign', 'violence', 'violence']
    })
    
    weights = compute_sample_weights(
        df, 
        label_column='label', 
        weight_column='subcategory',
        use_label_weights=True,
        use_subcategory_weights=True
    )
    
    # On perfectly balanced data, all weights should be equal (all close to 1.0)
    assert len(weights) == len(df)
    # All weights should be the same (variance close to 0)
    assert weights.std() < 0.01
    assert abs(weights.mean() - 1.0) < 0.01
    
    print(f"✓ Balanced dataset weights: all equal to {weights[0]:.3f}")


def test_label_weights_only():
    """Test with only label weighting enabled."""
    df = pd.DataFrame({
        'label': [0, 0, 0, 1, 1, 1, 0, 1, 1, 1],  # Imbalanced: 4 vs 6
        'subcategory': ['benign'] * 10  # All same subcategory
    })
    
    weights = compute_sample_weights(
        df,
        label_column='label',
        weight_column='subcategory',
        use_label_weights=True,
        use_subcategory_weights=False
    )
    
    # Minority class (label=0) should have higher weight
    minority_weight = weights[0]
    majority_weight = weights[3]
    assert minority_weight > majority_weight
    
    print(f"✓ Label-only weights: minority={minority_weight:.3f}, majority={majority_weight:.3f}")


def test_subcategory_weights_only():
    """Test with only subcategory weighting enabled."""
    df = pd.DataFrame({
        'label': [0] * 10,  # All same label
        'subcategory': ['benign', 'benign', 'benign', 'benign', 'violence', 
                       'violence', 'sexual', 'sexual', 'sexual', 'sexual']
    })
    
    weights = compute_sample_weights(
        df,
        label_column='label',
        weight_column='subcategory',
        use_label_weights=False,
        use_subcategory_weights=True
    )
    
    # Violence (2 samples) should have higher weight than benign (4 samples)
    violence_weight = weights[4]
    benign_weight = weights[0]
    assert violence_weight > benign_weight
    
    print(f"✓ Subcategory-only weights: violence={violence_weight:.3f}, benign={benign_weight:.3f}")


def test_no_weights():
    """Test with no weighting enabled."""
    df = pd.DataFrame({
        'label': [0, 0, 0, 1, 1, 1, 0, 1, 1, 1],
        'subcategory': ['benign', 'benign', 'violence', 'violence', 'sexual', 
                       'sexual', 'benign', 'violence', 'sexual', 'sexual']
    })
    
    weights = compute_sample_weights(
        df,
        label_column='label',
        weight_column='subcategory',
        use_label_weights=False,
        use_subcategory_weights=False
    )
    
    # All weights should be 1.0
    assert np.allclose(weights, 1.0)
    
    print(f"✓ No weighting: all weights = {weights[0]:.3f}")


if __name__ == "__main__":
    test_compute_class_weights()
    test_class_weights_balanced()
    test_label_weights_only()
    test_subcategory_weights_only()
    test_no_weights()
    print("\n✓ All class weight tests passed!")
