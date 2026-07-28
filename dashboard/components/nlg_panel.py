"""Natural language explanation panel component."""
import streamlit as st


def render(narrative: str, top_features: list, risk_colour: str):
    """Render the NLG explanation panel."""
    st.markdown("#### 📝 Plain-Language Explanation")
    bullets = "\n".join(
        f"- **{f['feature']}** (SHAP importance: {f['shap']:+.2f})" for f in top_features
    )
    st.markdown(
        f"""<div style="border-left:6px solid {risk_colour};
        background:#fafafa;padding:12px;border-radius:6px">
        <b>Explanation</b><br>{bullets}
        <br><br><b>Generated Narrative</b><br>{narrative}
        </div>""",
        unsafe_allow_html=True,
    )