"""Common constants and shared values for the intention_jailbreak package."""

# Dataset stratification columns
# Used consistently across train/test and train/val splits
STRATIFICATION_COLUMNS = ['prompt_harm_label', 'adversarial', 'subcategory']

# Label columns
LABEL_COLUMN = 'prompt_harm_label'
TEXT_COLUMN = 'prompt'
POSITIVE_LABEL = 'harmful'
