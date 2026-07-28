"""Feature selection using L1 regularisation for diagnostic purposes."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from code.data_prep.encode_features import (
    build_preprocessor, ORDINAL_COLS_WITH_ORDER, NUMERIC_COLS, NOMINAL_COLS,
)

# The selected feature set is fixed in config.py. L1 is used here only as
# a diagnostic to confirm the 18-feature set is well-supported.
L1_FEATURES = (
    NUMERIC_COLS
    + list(ORDINAL_COLS_WITH_ORDER.keys())
    + NOMINAL_COLS
)


def l1_diagnostic(X: pd.DataFrame, y: pd.Series, random_state: int = 0):
    """
    Returns L1 diagnostic information: number of non-zero coefficients,
    feature importance order, and coefficient values.
    
    This is a diagnostic tool to validate that the 18 selected features
    have strong predictive signal. The actual feature selection is
    fixed in config.py to ensure reproducibility.
    """
    pre = build_preprocessor()
    Xt = pre.fit_transform(X)
    Xt = StandardScaler().fit_transform(Xt)
    model = LogisticRegression(
        penalty="l1", solver="liblinear", C=0.1, max_iter=2000, random_state=random_state
    )
    model.fit(Xt, y)
    coefs = np.abs(model.coef_).mean(axis=0)
    order = np.argsort(coefs)[::-1]
    return int((coefs > 0).sum()), order, coefs
