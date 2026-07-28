"""Tests for model training and evaluation."""
import pytest
import numpy as np
from sklearn.metrics import accuracy_score
from src.data_prep.load_data import load_raw
from src.data_prep.clean_data import clean, NUMERIC_COLS, CATEGORICAL_COLS
from src.data_prep.encode_features import encode_target
from src.data_prep.split_data import split
from src.models.train_logreg import build_logreg_pipeline
from src.models.train_rf import build_rf_pipeline
from src.models.ordinal_metrics import quadratic_weighted_kappa, within_one_class_accuracy


def test_logreg_above_chance():
    df = clean(load_raw())
    X = df[CATEGORICAL_COLS + NUMERIC_COLS].copy()
    y = encode_target(df["Risk_Level"])
    Xtr, _, _, ytr, _, _ = split(X, y)
    pipe = build_logreg_pipeline()
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xtr)
    assert accuracy_score(ytr, pred) > 0.35  # above 0.25 chance for 4-class


def test_rf_above_chance():
    df = clean(load_raw())
    X = df[CATEGORICAL_COLS + NUMERIC_COLS].copy()
    y = encode_target(df["Risk_Level"])
    Xtr, _, _, ytr, _, _ = split(X, y)
    pipe = build_rf_pipeline(n_estimators=50)
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xtr)
    assert accuracy_score(ytr, pred) > 0.35


def test_qwk_perfect_agreement():
    y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    y_pred = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    assert quadratic_weighted_kappa(y_true, y_pred) == pytest.approx(1.0, abs=0.01)


def test_within_one_perfect():
    y_true = np.array([0, 1, 2, 3])
    y_pred = np.array([0, 1, 2, 3])
    assert within_one_class_accuracy(y_true, y_pred) == 1.0


def test_within_one_off_by_one():
    y_true = np.array([0, 1, 2, 3])
    y_pred = np.array([1, 2, 3, 2])  # all within one
    assert within_one_class_accuracy(y_true, y_pred) == 1.0


def test_within_one_off_by_two():
    y_true = np.array([0, 1, 2, 3])
    y_pred = np.array([2, 3, 0, 1])  # off by 2-3
    assert within_one_class_accuracy(y_true, y_pred) < 1.0