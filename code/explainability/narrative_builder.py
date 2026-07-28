"""Natural-language explanation builder for risk predictions."""


def build_narrative(head_class: str, top_features: list, proba: dict) -> str:
    """Build a plain-language explanation for a risk prediction."""
    prob_str = ", ".join(f"P({k})={v:.2f}" for k, v in proba.items())
    body_lines = []
    for name, val in top_features:
        direction = "increased" if val > 0 else "decreased"
        body_lines.append(
            f"- **{name}** (SHAP importance: {val:+.2f}) {direction} the predicted risk."
        )
    body = "\n".join(body_lines)
    return (
        f"**Predicted Risk Level**: {head_class}\n\n"
        f"**Class probabilities**: {prob_str}\n\n"
        f"**Primary risk drivers**:\n{body}"
    )


def top_features_from_shap(head_shap: dict, k: int = 5) -> list:
    """Return the top-k features by absolute SHAP value."""
    return sorted(head_shap.items(), key=lambda kv: abs(kv[1]), reverse=True)[:k]