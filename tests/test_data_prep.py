"""Tests for data preparation modules."""
import pandas as pd
import pytest
from code.data_prep.load_data import load_raw
from code.data_prep.validate_data import validate_data
from code.data_prep.clean_data import clean, NUMERIC_COLS, CATEGORICAL_COLS
from code.data_prep.encode_features import encode_target, build_preprocessor, TARGET_ORDER
from code.data_prep.split_data import split


def test_load_raw_ok():
    df = load_raw()
    assert len(df) > 0
    assert "Risk_Level" in df.columns


def test_validate_catches_missing_required_field():
    df = load_raw().drop(columns=["Risk_Level"])
    errors = validate_data(df)
    assert len(errors) > 0


def test_clean_returns_four_levels():
    df = clean(load_raw())
    assert set(df["Risk_Level"].unique()) == {"Low", "Medium", "High", "Critical"}


def test_clean_critical_count_at_least_one():
    df = clean(load_raw())
    assert (df["Risk_Level"] == "Critical").sum() > 0


def test_encode_target_ordinal():
    y = encode_target(pd.Series(["Low", "Medium", "High", "Critical"]))
    assert list(y) == [0, 1, 2, 3]


def test_split_preserves_class_proportions():
    df = clean(load_raw())
    X = df[CATEGORICAL_COLS + NUMERIC_COLS].copy()
    y = encode_target(df["Risk_Level"])
    Xtr, Xv, Xt, ytr, yv, yt = split(X, y)
    full_ratios = y.value_counts(normalize=True).sort_index()
    for part_name, part in [("train", ytr), ("val", yv), ("test", yt)]:
        part_ratios = part.value_counts(normalize=True).sort_index()
        for cls in full_ratios.index:
            assert abs(part_ratios.get(cls, 0) - full_ratios.get(cls, 0)) < 0.1


def test_build_preprocessor_fits():
    df = clean(load_raw())
    pre = build_preprocessor()
    X = df[CATEGORICAL_COLS + NUMERIC_COLS].iloc[:200]
    Xt = pre.fit_transform(X)
    assert Xt.shape[0] == 200
    assert Xt.shape[1] > 0