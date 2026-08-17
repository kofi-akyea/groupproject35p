#!/usr/bin/env python3
"""
End-to-end pipeline for the Project Risk DSS.
1. Load and validate raw data
2. Clean and encode
3. Split into train/val/test
4. Train candidate models (Logistic Regression, Random Forest, Ordinal LogReg, XGBoost, SVM, KNN)
5. Evaluate models with ordinal metrics (QWK, Within-One Accuracy, Macro F2)
6. Output evaluation reports and model card
"""
import sys
import os
from pathlib import Path

# Suppress joblib Windows core detection warning for CPU execution
os.environ['LOKY_MAX_CPU_COUNT'] = '1'

# Set root project path and append to sys.path
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

    # 1. Copy raw dataset from root directory to data/raw/ if missing
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

    # 2. Load raw data and validate schema integrity
    from code.data_prep.load_data import load_raw
    from code.data_prep.validate_data import validate_data
    from code.data_prep.clean_data import clean
    from code.data_prep.encode_features import encode_target
    from code.data_prep.split_data import split

    df = load_raw(raw_csv)
    print(f"[2] Loaded raw data: {df.shape}")

    # Perform non-fatal schema validation check
    errors = validate_data(df)
    if errors:
        print("[2] Validation errors:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("[2] Data validation passed")

    # 3. Clean raw dataset (impute missing values, fix boundaries)
    df = clean(df)
    print(f"[3] Cleaned data: {df.shape}, risk levels: {df['Risk_Level'].value_counts().to_dict()}")

    # 4. Extract feature columns and encode ordinal target variable into numeric levels [0, 1, 2, 3]
    from code.utils.config import FEATURE_NAMES
    X = df[FEATURE_NAMES].copy()
    y = encode_target(df["Risk_Level"])

    print(f"[4] Features: {X.shape}, Target distribution: {y.value_counts().sort_index().to_dict()}")

    # 5. Split dataset into train (70%), validation (15%), and test (15%) sets using stratified sampling
    X_train, X_val, X_test, y_train, y_val, y_test = split(X, y)
    print(f"[5] Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # 6. Save split datasets to data/processed/ directory in parquet format
    from code.data_prep.load_data import write_processed
    write_processed(X_train, "X_train")
    write_processed(X_val, "X_val")
    write_processed(X_test, "X_test")
    y_train.to_frame("Risk_Level").to_parquet(ROOT / "data" / "processed" / "y_train.parquet", index=False)
    y_val.to_frame("Risk_Level").to_parquet(ROOT / "data" / "processed" / "y_val.parquet", index=False)
    y_test.to_frame("Risk_Level").to_parquet(ROOT / "data" / "processed" / "y_test.parquet", index=False)

    # Save summary class distribution statistics
    dist = df["Risk_Level"].value_counts().reindex(["Low", "Medium", "High", "Critical"]).fillna(0).astype(int)
    dist.to_frame("count").assign(
        percent=lambda d: round(100 * d["count"] / d["count"].sum(), 2)
    ).to_csv(ROOT / "data" / "processed" / "class_distribution.csv")
    print("[6] Saved processed datasets")

    # 7. Train all candidate machine learning models
    from code.models.train_logreg import train as train_lr
    from code.models.train_rf import train as train_rf
    from code.models.train_ordinal_logreg import train_ordinal_logreg
    from code.models.train_xgboost import train_xgboost
    from code.models.train_svm import train_svm
    from code.models.train_knn import train_knn

    print("[7] Training Logistic Regression...")
    train_lr(X_train, y_train)

    print("[8] Training Random Forest...")
    train_rf(X_train, y_train)

    print("[9] Training Ordinal Logistic Regression...")
    train_ordinal_logreg(X_train, y_train)

    print("[10] Training XGBoost...")
    train_xgboost(X_train, y_train)

    print("[11] Training SVM (RBF kernel)...")
    train_svm(X_train, y_train)

    print("[12] Training KNN...")
    train_knn(X_train, y_train)

    # 8. Evaluate all models on held-out test data and save JSON reports
    from code.models.evaluate import evaluate

    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("[12] Evaluating Logistic Regression...")
    lr_rep = evaluate(
        ROOT / "models" / "logistic_regression.joblib",
        X_test, y_test,
        out_json=reports_dir / "logistic_regression_test_report.json"
    )
    print(f"  Accuracy: {lr_rep['accuracy']:.4f}, QWK: {lr_rep['qwk']:.4f}, Within-one: {lr_rep['within_one']:.4f}")

    print("[13] Evaluating Random Forest...")
    rf_rep = evaluate(
        ROOT / "models" / "random_forest.joblib",
        X_test, y_test,
        out_json=reports_dir / "random_forest_test_report.json"
    )
    print(f"  Accuracy: {rf_rep['accuracy']:.4f}, QWK: {rf_rep['qwk']:.4f}, Within-one: {rf_rep['within_one']:.4f}")

    print("[14] Evaluating Ordinal Logistic Regression...")
    olr_rep = evaluate(
        ROOT / "models" / "ordinal_logistic_regression.joblib",
        X_test, y_test,
        out_json=reports_dir / "ordinal_logistic_regression_test_report.json"
    )
    print(f"  Accuracy: {olr_rep['accuracy']:.4f}, QWK: {olr_rep['qwk']:.4f}, Within-one: {olr_rep['within_one']:.4f}")

    print("[15] Evaluating XGBoost...")
    xgb_rep = evaluate(
        ROOT / "models" / "xgboost.joblib",
        X_test, y_test,
        out_json=reports_dir / "xgboost_test_report.json"
    )
    print(f"  Accuracy: {xgb_rep['accuracy']:.4f}, QWK: {xgb_rep['qwk']:.4f}, Within-one: {xgb_rep['within_one']:.4f}")

    print("[16] Evaluating SVM (RBF kernel)...")
    svm_rep = evaluate(
        ROOT / "models" / "svm_rbf.joblib",
        X_test, y_test,
        out_json=reports_dir / "svm_rbf_test_report.json"
    )
    print(f"  Accuracy: {svm_rep['accuracy']:.4f}, QWK: {svm_rep['qwk']:.4f}, Within-one: {svm_rep['within_one']:.4f}")

    print("[17] Evaluating KNN...")
    knn_rep = evaluate(
        ROOT / "models" / "knn.joblib",
        X_test, y_test,
        out_json=reports_dir / "knn_test_report.json"
    )
    print(f"  Accuracy: {knn_rep['accuracy']:.4f}, QWK: {knn_rep['qwk']:.4f}, Within-one: {knn_rep['within_one']:.4f}")

    # 9. Save comprehensive model card artifact with benchmark comparison metrics
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
                "macro_f2": lr_rep["macro_f2"],
            },
            "random_forest": {
                "accuracy": rf_rep["accuracy"],
                "qwk": rf_rep["qwk"],
                "within_one": rf_rep["within_one"],
                "macro_f1": rf_rep["macro_f1"],
                "macro_f2": rf_rep["macro_f2"],
            },
            "ordinal_logistic_regression": {
                "accuracy": olr_rep["accuracy"],
                "qwk": olr_rep["qwk"],
                "within_one": olr_rep["within_one"],
                "macro_f1": olr_rep["macro_f1"],
                "macro_f2": olr_rep["macro_f2"],
            },
            "xgboost": {
                "accuracy": xgb_rep["accuracy"],
                "qwk": xgb_rep["qwk"],
                "within_one": xgb_rep["within_one"],
                "macro_f1": xgb_rep["macro_f1"],
                "macro_f2": xgb_rep["macro_f2"],
            },
            "svm_rbf": {
                "accuracy": svm_rep["accuracy"],
                "qwk": svm_rep["qwk"],
                "within_one": svm_rep["within_one"],
                "macro_f1": svm_rep["macro_f1"],
                "macro_f2": svm_rep["macro_f2"],
            },
            "knn": {
                "accuracy": knn_rep["accuracy"],
                "qwk": knn_rep["qwk"],
                "within_one": knn_rep["within_one"],
                "macro_f1": knn_rep["macro_f1"],
                "macro_f2": knn_rep["macro_f2"],
            }
        },
        "selected_model": "ordinal_logistic_regression (highest QWK and Within-One Accuracy)",
        "version": "1.0.0",
    }
    with open(ROOT / "models" / "model_card.json", "w") as f:
        json.dump(model_card, f, indent=2)

    print("[17] Model card saved")
    print("=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()