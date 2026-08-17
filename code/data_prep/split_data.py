"""Train/val/test splitting utilities."""
import pandas as pd
from sklearn.model_selection import train_test_split

from code.data_prep.encode_features import TARGET_COL

# Proportion of total dataset reserved for evaluation test set (15%) and validation set (15%)
TEST_SIZE = 0.15
VAL_SIZE = 0.15


def split(X: pd.DataFrame, y: pd.Series, random_state: int = 0):
    """Stratified split into train/val/test sets."""
    # First split: reserve 70% for training set and 30% for temporary holdout
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=TEST_SIZE + VAL_SIZE, stratify=y, random_state=random_state
    )
    
    # Second split: partition 30% holdout equally into validation set (15%) and test set (15%)
    val_rel = VAL_SIZE / (TEST_SIZE + VAL_SIZE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1 - val_rel, stratify=y_temp, random_state=random_state
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test