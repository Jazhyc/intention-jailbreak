"""Data preparation utilities for classification tasks."""

import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split
from datasets import Dataset

from ..common import STRATIFICATION_COLUMNS


def prepare_classification_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    val_size: float = 0.1,
    label_column: str = 'prompt_harm_label',
    text_column: str = 'prompt',
    positive_label: str = 'harmful',
    random_state: int = 42
) -> Tuple[Dataset, Dataset]:
    """
    Prepare data for binary classification (train and val only, no test).
    
    Uses stratified split on STRATIFICATION_COLUMNS to maintain the same 
    distribution in train and validation sets.
    
    Args:
        train_df: Training dataframe
        test_df: Test dataframe (not used, kept for compatibility)
        val_size: Fraction of training data to use for validation
        label_column: Name of the column containing labels
        text_column: Name of the column containing text
        positive_label: Label value that should be encoded as 1 (positive class)
        random_state: Random state for reproducibility
    
    Returns:
        Tuple of (train_dataset, val_dataset) as HuggingFace Datasets
    """
    # Create combined stratification key using shared columns
    stratify_key = train_df[STRATIFICATION_COLUMNS].astype(str).agg('_'.join, axis=1)
    
    train_df, val_df = train_test_split(
        train_df,
        test_size=val_size,
        random_state=random_state,
        stratify=stratify_key
    )
    
    print(f"Train: {len(train_df)} examples")
    print(f"Validation: {len(val_df)} examples")
    
    def add_binary_labels(df):
        df = df.copy()
        df['label'] = (df[label_column] == positive_label).astype(int)
        return df
    
    train_df = add_binary_labels(train_df)
    val_df = add_binary_labels(val_df)
    
    print(f"Positive class ({positive_label}): Train {train_df['label'].mean()*100:.1f}%, Val {val_df['label'].mean()*100:.1f}%")
    
    train_dataset = Dataset.from_pandas(train_df[[text_column, 'label']])
    val_dataset = Dataset.from_pandas(val_df[[text_column, 'label']])
    
    return train_dataset, val_dataset


def tokenize_dataset(dataset: Dataset, tokenizer, max_length: int, text_column: str = 'prompt', num_proc: int = 1):
    """
    Tokenize a dataset.
    
    Args:
        dataset: HuggingFace Dataset to tokenize
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        text_column: Name of the text column to tokenize
        num_proc: Number of processes to use for parallel tokenization
    
    Returns:
        Tokenized dataset
    """
    def tokenize_function(examples):
        return tokenizer(
            examples[text_column],
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
    
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=[text_column],
        num_proc=num_proc,
        desc="Tokenizing"
    )
    
    return tokenized
