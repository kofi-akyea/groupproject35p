"""Ordinal-aware evaluation metrics for four-level risk classification."""
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
)
from sklearn.utils import column_or_1d


def quadratic_weighted_kappa(y_true, y_pred, labels=None) -> float:
    """Quadratic-weighted Cohen's Kappa for ordinal labels."""
    # Ensure input targets are 1D numpy arrays
    y_true = column_or_1d(np.asarray(y_true))
    y_pred = column_or_1d(np.asarray(y_pred))
    
    # Infer unique labels if not provided
    if labels is None:
        labels = sorted(set(np.concatenate([y_true, y_pred])))
    labels = list(labels)
    K = len(labels)
    
    # Compute observed confusion matrix O
    y_true_idx = np.searchsorted(labels, y_true)
    y_pred_idx = np.searchsorted(labels, y_pred)
    O = np.zeros((K, K), dtype=float)
    for i, j in zip(y_true_idx, y_pred_idx):
        O[i, j] += 1
    N = O.sum()
    
    # Construct quadratic distance penalty weight matrix W
    W = np.zeros((K, K), dtype=float)
    for i in range(K):
        for j in range(K):
            W[i, j] = ((i - j) ** 2) / float((K - 1) ** 2)
            
    # Compute expected outer product matrix E under chance agreement
    row_marg = O.sum(axis=1)
    col_marg = O.sum(axis=0)
    E = np.outer(row_marg, col_marg) / N
    
    # Compute QWK score formula: 1 - sum(W * O) / sum(W * E)
    num = (W * O).sum()
    den = (W * E).sum()
    return 1.0 - (num / den if den > 0 else 0.0)


def within_one_class_accuracy(y_true, y_pred) -> float:
    """Accuracy allowing predictions within one ordinal level."""
    # Calculate proportion of predictions within +/- 1 ordinal step of true class
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float((np.abs(y_true - y_pred) <= 1).mean())


def f_beta_score(y_true, y_pred, beta=2, average='macro') -> float:
    """
    Calculate F-beta score, which weights recall more heavily than precision.
    
    Formula:
    F_β = (1 + β²) * (precision * recall) / (β² * precision + recall)
    
    For β=2, recall is weighted twice as heavily as precision.
    This is appropriate for risk classification where missing high-risk cases
    (false negatives) is more costly than over-estimating risk (false positives).
    """
    # Calculate precision and recall
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    
    # Apply F-beta weighting formula
    beta_squared = beta ** 2
    numerator = (1 + beta_squared) * (precision * recall)
    denominator = (beta_squared * precision) + recall
    
    if denominator == 0:
        return 0.0
    
    return float(numerator / denominator)


def ordinal_report(y_true, y_pred, y_proba, class_names=("Low", "Medium", "High", "Critical")):
    """Generate comprehensive ordinal evaluation report."""
    # Aggregate accuracy, Macro F1, Macro F2, QWK, Within-One accuracy, and AUC metrics
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "macro_f2": float(f_beta_score(y_true, y_pred, beta=2, average='macro')),
        "qwk": float(quadratic_weighted_kappa(y_true, y_pred, labels=list(range(4)))),
        "within_one": within_one_class_accuracy(y_true, y_pred),
        "macro_auc_ovr": float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(4))).tolist(),
        "class_names": list(class_names),
    }