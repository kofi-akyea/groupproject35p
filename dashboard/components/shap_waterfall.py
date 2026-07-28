"""SHAP waterfall chart component."""
import matplotlib.pyplot as plt
import streamlit as st


def render(top_features, expected_value=None, predicted_value=None):
    """Render a custom waterfall chart of SHAP values."""
    feats = [f for f, _ in top_features][::-1]
    vals = [v for _, v in top_features][::-1]
    labels = feats
    running = expected_value if expected_value is not None else 0.0
    cum = [running]
    for v in vals:
        running += v
        cum.append(running)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    colors = ["#c0392b" if v > 0 else "#2ecc71" for v in vals]
    bottoms = [min(cum[i], cum[i + 1]) for i in range(len(feats))]
    ax.bar(range(len(feats)), [abs(v) for v in vals], bottom=bottoms, color=colors)
    ax.set_xticks(range(len(feats)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title("Local SHAP Contributions (Top Features)")
    ax.set_ylabel("Contribution to Log-Odds")
    plt.tight_layout()
    st.pyplot(fig)