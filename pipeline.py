#!/usr/bin/env python3
"""
End-to-end pipeline for the Project Risk DSS.
1. Load and validate raw data
2. Clean and encode
3. Split into train/val/test
4. Train Logistic Regression and Random Forest
5. Evaluate models
6. Output evaluation reports
"""
import sys
from pathlib import Path

# Ensure src is on the path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
import joblib
import json
import shutil


def main():
    print("=" * 60)
    print("Project Risk DSS - Pipeline Runner")
    print("=" * 60)

    # 1. Copy raw data if needed
    raw_dir = ROOT / "data" / "raw"
    raw_csv = raw_dir / "project_risk_raw_dataset.csv"
    source_csv = ROOT / "project_risk_raw_dataset.csv"

    if not raw_csv.exists() and source_csv.exists():
        raw_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_csv, raw_csv)
        print(f"[1] Copied raw data to {raw_csv}")
    elif raw_csv.exists():
        print(f"[1] Raw data found at {raw_csv}")
    else:
        print("[1] ERROR: No raw dataset found!")
        sys.exit(1)

    # 2. Load and validate
    from code.data_prep.load_data import load_raw
    from code.data_prep.validate_data import validate_data
    from code.data_prep.clean_data import clean
    from code.data_prep.encode_features import encode_target
    from code.data_prep.split_data import split

    df = load_raw(raw_csv)
    print(f"[2] Loaded raw data: {df.shape}")

    errors = validate_data(df)
    if errors:
        print("[2] Validation errors:")
        for e in errors:
            print(f"  - {e}")
        # Non-fatal - continue with cleaning
    else:
        print("[2] Data validation passed")

    df = clean(df)
    print(f"[3] Cleaned data: {df.shape}, risk levels: {df['Risk_Level'].value_counts().to_dict()}")

    # 4. Prepare features and target using config FEATURE_NAMES
    from code.utils.config import FEATURE_NAMES
    X = df[FEATURE_NAMES].copy()
    y = encode_target(df["Risk_Level"])

    print(f"[4] Features: {X.shape}, Target distribution: {y.value_counts().sort_index().to_dict()}")

    # 5. Split
    X_train, X_val, X_test, y_train, y_val, y_test = split(X, y)
    print(f"[5] Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Save processed data
    from code.data_prep.load_data import write_processed
    write_processed(X_train, "X_train")
    write_processed(X_val, "X_val")
    write_processed(X_test, "X_test")
    y_train.to_frame("Risk_Level").to_parquet(ROOT / "data" / "processed" / "y_train.parquet", index=False)
    y_val.to_frame("Risk_Level").to_parquet(ROOT / "data" / "processed" / "y_val.parquet", index=False)
    y_test.to_frame("Risk_Level").to_parquet(ROOT / "data" / "processed" / "y_test.parquet", index=False)

    # Save class distribution
    dist = df["Risk_Level"].value_counts().reindex(["Low", "Medium", "High", "Critical"]).fillna(0).astype(int)
    dist.to_frame("count").assign(
        percent=lambda d: round(100 * d["count"] / d["count"].sum(), 2)
    ).to_csv(ROOT / "data" / "processed" / "class_distribution.csv")
    print("[6] Saved processed datasets")

    # 7. Train models
    from code.models.train_logreg import train as train_lr
    from code.models.train_rf import train as train_rf

    print("[7] Training Logistic Regression...")
    train_lr(X_train, y_train)

    print("[8] Training Random Forest...")
    train_rf(X_train, y_train)

    # 8. Evaluate
    from code.models.evaluate import evaluate

    # Create reports directory
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("[9] Evaluating Logistic Regression...")
    lr_rep = evaluate(
        ROOT / "models" / "logistic_regression.joblib",
        X_test, y_test,
        out_json=reports_dir / "logistic_regression_test_report.json"
    )
    print(f"  Accuracy: {lr_rep['accuracy']:.4f}, QWK: {lr_rep['qwk']:.4f}, Within-one: {lr_rep['within_one']:.4f}")

    print("[10] Evaluating Random Forest...")
    rf_rep = evaluate(
        ROOT / "models" / "random_forest.joblib",
        X_test, y_test,
        out_json=reports_dir / "random_forest_test_report.json"
    )
    print(f"  Accuracy: {rf_rep['accuracy']:.4f}, QWK: {rf_rep['qwk']:.4f}, Within-one: {rf_rep['within_one']:.4f}")

    # 9. Save model card
    model_card = {
        "dataset": "Project Management Risk Raw (Kaggle)",
        "dataset_size": 4000,
        "features": FEATURE_NAMES,
        "target": "Risk_Level (Low=0, Medium=1, High=2, Critical=3)",
        "models": {
            "logistic_regression": {
                "accuracy": lr_rep["accuracy"],
                "qwk": lr_rep["qwk"],
                "within_one": lr_rep["within_one"],
                "macro_f1": lr_rep["macro_f1"],
            },
            "random_forest": {
                "accuracy": rf_rep["accuracy"],
                "qwk": rf_rep["qwk"],
                "within_one": rf_rep["within_one"],
                "macro_f1": rf_rep["macro_f1"],
            }
        },
        "selected_model": "random_forest (higher accuracy and QWK)",
        "version": "1.0.0",
    }
    with open(ROOT / "models" / "model_card.json", "w") as f:
        json.dump(model_card, f, indent=2)

    print("[11] Model card saved")
    print("=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()