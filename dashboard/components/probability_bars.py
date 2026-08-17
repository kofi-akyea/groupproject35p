"""Probability bars component."""
import streamlit as st
from code.utils.risk_levels import RISK_COLOURS


def render(probabilities: dict):
    """Render probability bars for each risk class."""
    st.markdown("<h4 style='margin: 16px 0 12px 0; font-size: 16px; font-weight: 600;'>Class Probabilities</h4>", unsafe_allow_html=True)
    
    # Iterate through predicted class probabilities and render progress bars
    for k, v in probabilities.items():
        # Clamp probability value to valid [0.0, 1.0] float range for Streamlit progress widget
        progress_value = min(max(float(v) / 100, 0.0), 1.0)
        st.progress(progress_value, text=f"{k}: {float(v):.1f}%")