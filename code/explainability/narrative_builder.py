"""Natural-language explanation builder for risk predictions."""


def build_narrative(head_class: str, top_features: list, proba: dict) -> str:
    """Build a plain-language explanation for a risk prediction."""
    # Format class probabilities as string
    prob_str = ", ".join(f"P({k})={v:.2f}%" for k, v in proba.items())
    body_lines = []
    
    # Construct explanation sentence for each key feature driver
    for name, val in top_features:
        direction = "increased" if val > 0 else "decreased"
        shap_str = f"{val:+.3f}"
        body_lines.append(
            f"- **{name}** (SHAP importance: {shap_str}) {direction} the predicted risk."
        )
    body = "\n".join(body_lines)
    
    # Return formatted HTML block containing headline prediction and driver narrative
    return (
        f"<div style='background-color: #F8FAFC; border: 1px solid #E2E8F0; color: #0F172A; padding: 12px 16px; border-radius: 8px; margin-bottom: 16px;'>"
        f"<strong>Predicted Risk Level: {head_class}</strong></div>\n\n"
        f"**Class probabilities**: {prob_str}\n\n"
        f"**Primary risk drivers**:\n{body}"
    )


def top_features_from_shap(head_shap: dict, k: int = 5) -> list:
    """Return the top-k features by absolute SHAP value."""
    # Sort features in descending order of absolute SHAP magnitude
    return sorted(head_shap.items(), key=lambda kv: abs(kv[1]), reverse=True)[:k]