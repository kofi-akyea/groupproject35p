"""SHAP-based explanation engine for model predictions."""
import joblib
import numpy as np
import pandas as pd
import shap

from code.data_prep.encode_features import (
    NUMERIC_COLS, NOMINAL_COLS, ORDINAL_COLS_WITH_ORDER
)

CLASS_NAMES = ["Low", "Medium", "High", "Critical"]


def build_feature_origin_map(preprocessor) -> list:
    """Return one original feature name per transformed column."""
    origins = []
    pre = preprocessor

    # Numeric passthrough block
    for name in pre.transformers_[0][2]:
        origins.append(name)

    # Ordinal block
    for name in pre.transformers_[1][2]:
        origins.append(name)

    # One-hot block
    ohe = pre.transformers_[2][1]
    nominal_cols = pre.transformers_[2][2]
    cat_iter = ohe.categories_
    for col, cats in zip(nominal_cols, cat_iter):
        for _ in cats:
            origins.append(col)
    return origins


def _aggregate_origin(shap_values_row: np.ndarray, origins: list) -> dict:
    """Sum per-column SHAP values back to per-feature SHAP values."""
    agg = {}
    for v, src in zip(shap_values_row, origins):
        # Handle both scalar and array SHAP values
        if np.isscalar(v):
            val = float(v)
        else:
            val = float(np.sum(v))
        agg[src] = agg.get(src, 0.0) + val
    return agg


def shap_for_pipeline(pipeline, X_row: pd.DataFrame) -> dict:
    """Return per-class probabilities and per-feature SHAP values."""
    pre = pipeline.named_steps["pre"]
    clf = pipeline.named_steps["clf"]
    Xt = pre.transform(X_row)

    if hasattr(clf, "estimators_"):
        # Tree-based model (Random Forest)
        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(Xt)
        origins = build_feature_origin_map(pre)
        probs = clf.predict_proba(Xt)[0]
        per_class = {CLASS_NAMES[k]: float(probs[k]) for k in range(len(CLASS_NAMES))}
        per_class_shap = {}
        for k in range(len(CLASS_NAMES)):
            # Handle empty SHAP values
            if isinstance(sv, list):
                shap_row = sv[k][0] if len(sv) > k and len(sv[k]) > 0 else np.zeros(len(origins))
            else:
                shap_row = sv[0] if len(sv) > 0 else np.zeros(len(origins))
            per_class_shap[CLASS_NAMES[k]] = _aggregate_origin(shap_row, origins)
        head_idx = int(np.argmax(probs))
        head = CLASS_NAMES[head_idx]
        return {
            "probabilities": per_class,
            "head": head,
            "head_shap": per_class_shap[head],
            "head_class_index": head_idx,
        }
    else:
        # Linear model
        explainer = shap.LinearExplainer(clf, masker=shap.maskers.Impute(Xt, method="linear"))
        sv = explainer.shap_values(Xt)
        origins = build_feature_origin_map(pre)
        probs = clf.predict_proba(Xt)[0]
        per_class = {CLASS_NAMES[k]: float(probs[k]) for k in range(len(CLASS_NAMES))}
        per_class_shap = {}
        for k in range(len(CLASS_NAMES)):
            # Handle empty SHAP values
            if isinstance(sv, list):
                shap_row = sv[k] if len(sv) > k else np.zeros(len(origins))
            else:
                shap_row = sv if sv is not None and len(sv) > 0 else np.zeros(len(origins))
            per_class_shap[CLASS_NAMES[k]] = _aggregate_origin(shap_row, origins)
        head_idx = int(np.argmax(probs))
        head = CLASS_NAMES[head_idx]
        return {
            "probabilities": per_class,
            "head": head,
            "head_shap": per_class_shap[head],
            "head_class_index": head_idx,
        }