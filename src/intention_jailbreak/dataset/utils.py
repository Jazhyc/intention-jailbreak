"""Utility functions for dataset loading and preprocessing."""

from typing import Tuple, Optional
import pandas as pd
from datasets import load_dataset, DatasetDict
from sklearn.model_selection import train_test_split


def load_wildguardmix(subset: str = 'wildguardtrain') -> DatasetDict:
    """
    Load the WildGuardMix dataset from HuggingFace.
    
    Args:
        subset: The subset of the dataset to load. Default is 'wildguardtrain'.
    
    Returns:
        DatasetDict containing the loaded dataset splits.
    """
    print(f"Loading WildGuardMix dataset (subset: {subset})...")
    dataset = load_dataset("allenai/wildguardmix", subset)
    print(f"Dataset loaded successfully!")
    return dataset


def create_stratified_split(
    df: pd.DataFrame,
    test_size: float = 0.4,
    random_state: int = 42,
    stratify_columns: Optional[list] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create a stratified train/test split of the dataset.
    
    Args:
        df: The dataframe to split.
        test_size: The proportion of the dataset to include in the test split.
        random_state: Random state for reproducibility.
        stratify_columns: List of column names to stratify on. 
                         Default is ['prompt_harm_label', 'adversarial', 'subcategory'].
    
    Returns:
        Tuple of (train_df, test_df)
    """
    if stratify_columns is None:
        stratify_columns = ['prompt_harm_label', 'adversarial', 'subcategory']
    
    # Create a combined stratification column
    df_copy = df.copy()
    stratify_values = df_copy[stratify_columns].astype(str).agg('_'.join, axis=1)
    
    print(f"Creating {(1-test_size)*100:.0f}:{test_size*100:.0f} stratified split...")
    print(f"Stratifying on: {', '.join(stratify_columns)}")
    
    # Perform stratified split
    train_df, test_df = train_test_split(
        df_copy,
        test_size=test_size,
        stratify=stratify_values,
        random_state=random_state
    )
    
    print(f"Train set size: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Test set size: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    
    return train_df, test_df


def compare_distributions(
    original_df: pd.DataFrame,
    train_df: pd.DataFrame,
    columns: Optional[list] = None
) -> pd.DataFrame:
    """
    Compare distributions between original and train datasets.
    
    Args:
        original_df: The original dataset.
        train_df: The training split.
        columns: List of columns to compare. 
                Default is ['prompt_harm_label', 'response_harm_label', 'adversarial', 'subcategory'].
    
    Returns:
        DataFrame with comparison statistics.
    """
    if columns is None:
        columns = ['prompt_harm_label', 'response_harm_label', 'adversarial', 'subcategory']
    
    comparisons = {}
    
    for col in columns:
        if col not in original_df.columns:
            print(f"Warning: Column '{col}' not found in dataframe. Skipping.")
            continue
            
        orig_dist = original_df[col].value_counts(normalize=True).sort_index() * 100
        train_dist = train_df[col].value_counts(normalize=True).sort_index() * 100
        
        comparison = pd.DataFrame({
            'Original (%)': orig_dist,
            'Train (%)': train_dist,
            'Difference (%)': train_dist - orig_dist
        })
        
        comparisons[col] = comparison
    
    return comparisons


def get_dataset_statistics(df: pd.DataFrame) -> dict:
    """
    Get basic statistics about the dataset.
    
    Args:
        df: The dataframe to analyze.
    
    Returns:
        Dictionary containing dataset statistics.
    """
    stats = {
        'total_samples': len(df),
        'num_columns': len(df.columns),
        'columns': df.columns.tolist(),
        'prompt_harm_distribution': df['prompt_harm_label'].value_counts().to_dict(),
        'response_harm_distribution': df['response_harm_label'].value_counts().to_dict(),
        'adversarial_distribution': df['adversarial'].value_counts().to_dict(),
        'num_subcategories': df['subcategory'].nunique(),
        'top_10_subcategories': df['subcategory'].value_counts().head(10).to_dict(),
        'missing_values': df.isnull().sum().to_dict()
    }
    
    return stats
