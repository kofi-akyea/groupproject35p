"""Ordinal-aware evaluation metrics for four-level risk classification."""
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix
)
from sklearn.utils import column_or_1d


def quadratic_weighted_kappa(y_true, y_pred, labels=None) -> float:
    """Quadratic-weighted Cohen's Kappa for ordinal labels."""
    y_true = column_or_1d(np.asarray(y_true))
    y_pred = column_or_1d(np.asarray(y_pred))
    if labels is None:
        labels = sorted(set(np.concatenate([y_true, y_pred])))
    labels = list(labels)
    K = len(labels)
    y_true_idx = np.searchsorted(labels, y_true)
    y_pred_idx = np.searchsorted(labels, y_pred)
    O = np.zeros((K, K), dtype=float)
    for i, j in zip(y_true_idx, y_pred_idx):
        O[i, j] += 1
    N = O.sum()
    W = np.zeros((K, K), dtype=float)
    for i in range(K):
        for j in range(K):
            W[i, j] = ((i - j) ** 2) / float((K - 1) ** 2)
    row_marg = O.sum(axis=1)
    col_marg = O.sum(axis=0)
    E = np.outer(row_marg, col_marg) / N
    num = (W * O).sum()
    den = (W * E).sum()
    return 1.0 - (num / den if den > 0 else 0.0)


def within_one_class_accuracy(y_true, y_pred) -> float:
    """Accuracy allowing predictions within one ordinal level."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float((np.abs(y_true - y_pred) <= 1).mean())


def ordinal_report(y_true, y_pred, y_proba, class_names=("Low", "Medium", "High", "Critical")):
    """Generate comprehensive ordinal evaluation report."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "qwk": float(quadratic_weighted_kappa(y_true, y_pred, labels=list(range(4)))),
        "within_one": within_one_class_accuracy(y_true, y_pred),
        "macro_auc_ovr": float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(4))).tolist(),
        "class_names": list(class_names),
    }