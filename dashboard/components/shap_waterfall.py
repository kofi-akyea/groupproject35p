"""SHAP waterfall chart component."""
import matplotlib.pyplot as plt
import streamlit as st


def render(top_features, expected_value=None, predicted_value=None):
    """Render a custom waterfall chart of SHAP values."""
    # Reverse feature list so top features display in intuitive order
    feats = [f for f, _ in top_features][::-1]
    vals = [v for _, v in top_features][::-1]
    labels = feats
    
    # Calculate cumulative offsets to position floating waterfall bars
    running = expected_value if expected_value is not None else 0.0
    cum = [running]
    for v in vals:
        running += v
        cum.append(running)

    # Initialize figure canvas
    fig, ax = plt.subplots(figsize=(6, 3.8), dpi=100)
    
    # Set red for positive risk contributions and green for risk-reducing contributions
    colors = ["#DC2626" if v > 0 else "#16A34A" for v in vals]
    bottoms = [min(cum[i], cum[i + 1]) for i in range(len(feats))]
    
    # Draw waterfall contribution bars
    bars = ax.bar(range(len(feats)), [abs(v) for v in vals], bottom=bottoms, color=colors, width=0.5, edgecolor="none")
    ax.axhline(0, color="#94A3B8", linewidth=0.8, linestyle="--")
    
    # Configure axes labels and title styling
    ax.set_xticks(range(len(feats)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9, color="#334155")
    ax.set_title("Local Feature Contributions (SHAP)", fontsize=11, fontweight="600", color="#0F172A", pad=12)

    ax.set_ylabel("Impact on Risk Probability", fontsize=9, color="#64748B")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:+.2f}'))
    ax.tick_params(axis='both', which='major', labelsize=9, colors="#475569")
    
    # Remove top/right spines and add clean horizontal grid lines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#E2E8F0')
    ax.spines['bottom'].set_color('#E2E8F0')
    ax.grid(axis='y', linestyle=':', alpha=0.4, color='#CBD5E1')

    # Display plot inside Streamlit app
    plt.tight_layout()
    st.pyplot(fig)