"""Probability bars component."""
import streamlit as st
from code.utils.risk_levels import RISK_COLOURS


def render(probabilities: dict):
    """Render probability bars for each risk class."""
    st.markdown("#### Class Probabilities")
    for k, v in probabilities.items():
        colour = RISK_COLOURS.get(k, "#888")
        st.markdown(
            f"<span style='color:{colour};font-weight:600'>{k}</span>",
            unsafe_allow_html=True,
        )
        st.progress(min(max(float(v), 0.0), 1.0), text=f"{k}: {float(v):.1%}")