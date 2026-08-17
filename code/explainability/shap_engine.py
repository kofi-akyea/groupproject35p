"""SHAP-based explanation engine for model predictions."""
import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.preprocessing import OneHotEncoder

from code.data_prep.encode_features import (
    NUMERIC_COLS, NOMINAL_COLS, ORDINAL_COLS_WITH_ORDER
)

CLASS_NAMES = ["Low", "Medium", "High", "Critical"]


def build_feature_origin_map(preprocessor) -> list:
    """Return one original feature name per transformed column.

    Walks the fitted ColumnTransformer in its own declared order, so the map stays
    correct whichever order the numeric/ordinal/nominal blocks were registered in.
    One-hot blocks expand to one entry per category; every other transformer emits
    one entry per input column.
    """
    origins = []
    # Traverse fitted transformers in ColumnTransformer
    for name, transformer, cols in preprocessor.transformers_:
        if transformer == "drop" or name == "remainder":
            continue

        # Unwrap Pipeline step to access underlying encoder instance
        encoder = transformer
        if hasattr(transformer, "named_steps"):
            encoder = list(transformer.named_steps.values())[-1]

        # One-hot encoded features expand into multiple column origins
        if isinstance(encoder, OneHotEncoder):
            for col, cats in zip(cols, encoder.categories_):
                origins.extend([col] * len(cats))
        else:
            origins.extend(cols)
    return origins


def _class_shap_row(sv, class_index: int, n_cols: int) -> np.ndarray:
    """Extract the per-column SHAP vector for one class of the first sample.

    SHAP >= 0.45 returns a single array shaped (n_samples, n_columns, n_classes);
    older releases returned a list of (n_samples, n_columns) arrays, one per class.
    Both are handled so the explanation survives a library upgrade.
    """
    if isinstance(sv, list):
        if len(sv) > class_index and len(sv[class_index]) > 0:
            return np.asarray(sv[class_index][0])
        return np.zeros(n_cols)

    arr = np.asarray(sv)
    if arr.ndim == 3:                      # Shape: (samples, columns, classes)
        return arr[0, :, class_index]
    if arr.ndim == 2:                      # Shape: (samples, columns) - single output
        return arr[0]
    return np.zeros(n_cols)


def _aggregate_origin(shap_values_row: np.ndarray, origins: list) -> dict:
    """Sum per-column SHAP values back to per-feature SHAP values."""
    agg = {}
    # Aggregate one-hot expanded SHAP values back to raw feature name key
    for v, src in zip(shap_values_row, origins):
        if np.isscalar(v):
            val = float(v)
        else:
            val = float(np.sum(v))
        agg[src] = agg.get(src, 0.0) + val
    return agg


def shap_for_pipeline(pipeline, X_row: pd.DataFrame) -> dict:
    """Return per-class probabilities and per-feature SHAP values.
    
    IMPORTANT: SHAP explanation is only supported for Random Forest and XGBoost
    (tree-based models). Other models will raise an error if called.
    """
    from sklearn.ensemble import RandomForestClassifier
    try:
        import xgboost
        XGBClassifier = xgboost.XGBClassifier
    except ImportError:
        XGBClassifier = None
    
    # Extract preprocessing pipeline and classifier model
    pre = pipeline.named_steps["pre"]
    clf = pipeline.named_steps["clf"]
    Xt = pre.transform(X_row)

    # Verify classifier is a supported tree-based model
    is_tree_model = isinstance(clf, RandomForestClassifier)
    if XGBClassifier is not None:
        is_tree_model = is_tree_model or isinstance(clf, XGBClassifier)
    
    if not is_tree_model:
        raise ValueError(
            f"SHAP explanation is only supported for Random Forest and XGBoost. "
            f"Model type {type(clf).__name__} is not supported."
        )
    
    # Construct TreeExplainer and compute raw SHAP values
    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(Xt)
    origins = build_feature_origin_map(pre)
    probs = clf.predict_proba(Xt)[0]
    
    # Map SHAP values to class names and aggregate to feature origins
    per_class = {CLASS_NAMES[k]: float(probs[k]) for k in range(len(CLASS_NAMES))}
    per_class_shap = {}
    for k in range(len(CLASS_NAMES)):
        shap_row = _class_shap_row(sv, k, len(origins))
        per_class_shap[CLASS_NAMES[k]] = _aggregate_origin(shap_row, origins)
    
    head_idx = int(np.argmax(probs))
    head = CLASS_NAMES[head_idx]
    
    # Return structured output dictionary
    return {
        "probabilities": per_class,
        "head": head,
        "head_shap": per_class_shap[head],
        "head_class_index": head_idx,
    }