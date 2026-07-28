"""Model Performance page - Evaluation metrics and comparison."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import roc_curve, auc
from code.utils.config import PROCESSED_DIR, MODELS_DIR
from code.utils.risk_levels import RISK_LEVELS, RISK_COLOURS
from code.models.ordinal_metrics import ordinal_report
from code.data_prep.load_data import load_processed
from code.data_prep.encode_features import encode_target

st.set_page_config(page_title="Model Performance", page_icon="⚖️", layout="wide")
st.title("⚖️ Model Performance & Comparison")
st.caption("Ordinal-aware evaluation on the held-out test set.")

# Load test data
X_test = load_processed("X_test")
y_test = load_processed("y_test")["Risk_Level"]

# Load models
model_names = ["logistic_regression", "random_forest"]
model_labels = {"logistic_regression": "Logistic Regression", "random_forest": "Random Forest"}
results = {}

for name in model_names:
    model_path = MODELS_DIR / f"{name}.joblib"
    if model_path.exists():
        pipe = joblib.load(model_path)
        proba = pipe.predict_proba(X_test)
        pred = proba.argmax(axis=1)
        rep = ordinal_report(y_test, pred, proba, RISK_LEVELS)
        results[name] = rep

# Display metrics
if results:
    st.subheader("Key Metrics Comparison")
    cols = st.columns(4)
    metrics = ["accuracy", "within_one", "qwk", "macro_f1"]
    labels = ["Exact-Match Accuracy", "Within-One Accuracy", "QWK", "Macro F1"]

    for i, (metric, label) in enumerate(zip(metrics, labels)):
        with cols[i]:
            for name in model_names:
                if name in results:
                    val = results[name][metric]
                    st.metric(f"{label} ({model_labels[name]})", f"{val:.4f}")

    # Confusion matrix for best model
    best_model = "random_forest"
    if best_model in results:
        st.subheader(f"Confusion Matrix - {model_labels[best_model]} (Normalised)")
        cm = np.array(results[best_model]["confusion_matrix"])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm_norm, annot=True, fmt=".2f", xticklabels=RISK_LEVELS,
                    yticklabels=RISK_LEVELS, cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        plt.tight_layout()
        st.pyplot(fig)

    # ROC curves
    st.subheader("Macro ROC Curves (One-vs-Rest)")
    fig, ax = plt.subplots(figsize=(6, 5))
    for name in model_names:
        if name in results:
            pipe = joblib.load(MODELS_DIR / f"{name}.joblib")
            proba = pipe.predict_proba(X_test)
            for k, cls in enumerate(RISK_LEVELS):
                fpr, tpr, _ = roc_curve((y_test == k).astype(int), proba[:, k])
                ax.plot(fpr, tpr, color=RISK_COLOURS[cls],
                        label=f"{model_labels[name]} - {cls} (AUC={auc(fpr, tpr):.3f})",
                        linestyle="--" if "logistic" in name else "-")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    plt.tight_layout()
    st.pyplot(fig)

    # Per-class metrics
    st.subheader("Per-Class Performance (Random Forest)")
    if "per_class_report" in results.get("random_forest", {}):
        per_class_df = pd.DataFrame(results["random_forest"]["per_class_report"]).T
        st.dataframe(per_class_df.style.format("{:.3f}"))
else:
    st.warning("No trained models found. Run the pipeline first.")