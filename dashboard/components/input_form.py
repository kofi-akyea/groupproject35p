"""Project input form component for Streamlit."""
import streamlit as st
from code.utils.config import FEATURE_SPECS


def render():
    """Render the project input form and return a dict of values."""
    st.markdown("<h3 style='margin: 0 0 12px 0; font-size: 18px; font-weight: 600;'>Project Configuration</h3>", unsafe_allow_html=True)
    values = {}
    
    # Group features logically by category (e.g. Project, Team, Technical)
    groups = sorted({f.group for f in FEATURE_SPECS})
    for g in groups:
        with st.expander(g, expanded=(g == "Project")):
            for f in FEATURE_SPECS:
                if f.group != g:
                    continue
                
                # Render numeric features as sliders
                if f.kind == "numeric":
                    min_val = float(f.minv) if f.minv is not None else 0.0
                    max_val = float(f.maxv) if f.maxv is not None else 1.0
                    step_val = float(f.step) if f.step is not None else 0.01
                    default_val = float(f.default) if f.default is not None else min_val
                    values[f.name] = st.slider(
                        f.display, min_value=min_val, max_value=max_val,
                        value=default_val, step=step_val, key=f.name
                    )
                # Render ordinal features as ordered select sliders
                elif f.kind == "ordinal":
                    values[f.name] = st.select_slider(
                        f.display, options=f.order, value=f.default, key=f.name
                    )
                # Render nominal features as drop-down select boxes
                elif f.kind == "nominal":
                    idx = (f.order or []).index(f.default) if f.default in (f.order or []) else 0
                    values[f.name] = st.selectbox(
                        f.display, options=f.order, index=idx, key=f.name
                    )
    return values