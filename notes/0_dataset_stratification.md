# WildGuardMix Dataset Stratification

## Overview

This document describes the stratification methodology used to split the WildGuardMix dataset into training and test sets.

## Split Ratio

- **Training Set**: 60%
- **Test Set** (For later annotation): 40%

## Stratification Strategy

The dataset was split using stratified sampling to ensure that both training and test sets maintain the same proportions as the original dataset. Stratification was performed based on the following three variables:

1. **`prompt_harm_label`** - Whether the prompt itself is harmful or unharmful
2. **`adversarial`** - Boolean flag indicating if the prompt is adversarial / a jailbreak attempt
3. **`subcategory`** - The specific subcategory of the prompt/response pair

### Combined Stratification

To ensure proper representation across all three dimensions simultaneously, we created a combined stratification key by concatenating all three variables. This approach ensures that each unique combination of these three attributes is proportionally represented in both the training and test sets.

## Results

### Distribution Preservation

The stratified split successfully preserved the original dataset's distributions:

- **Prompt Harm Label**: Exact match between original and train distributions
- **Adversarial Flag**: Exact match between original and train distributions  
- **Subcategories**: All subcategory proportions maintained in the train set

### Response Harm Label (Not Stratified)

Although we did not explicitly stratify on `response_harm_label`, the distribution shift was minimal:

- **Maximum difference**: ~0.09%

This small shift indicates that the WildGuardMix dataset was already well-balanced with respect to response harm labels. The natural correlation between prompt characteristics (which we did stratify on) and response harm labels resulted in an inherently balanced split without explicit stratification.

## Reproducibility

- **Random State**: 42 (for reproducibility)
- **Library**: scikit-learn 1.7.2
