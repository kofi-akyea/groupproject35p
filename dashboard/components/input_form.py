"""Project input form component for Streamlit."""
import streamlit as st
from code.utils.config import FEATURE_SPECS


def render():
    """Render the project input form and return a dict of values."""
    st.markdown("### 🎛️ Project Input")
    values = {}
    groups = sorted({f.group for f in FEATURE_SPECS})
    for g in groups:
        with st.expander(g, expanded=(g == "Project")):
            for f in FEATURE_SPECS:
                if f.group != g:
                    continue
                if f.kind == "numeric":
                    # Ensure all slider parameters are the same type (float)
                    min_val = float(f.minv) if f.minv is not None else 0.0
                    max_val = float(f.maxv) if f.maxv is not None else 1.0
                    step_val = float(f.step) if f.step is not None else 0.01
                    default_val = float(f.default) if f.default is not None else min_val
                    values[f.name] = st.slider(
                        f.display, min_value=min_val, max_value=max_val,
                        value=default_val, step=step_val, key=f.name
                    )
                elif f.kind == "ordinal":
                    values[f.name] = st.select_slider(
                        f.display, options=f.order, value=f.default, key=f.name
                    )
                elif f.kind == "nominal":
                    idx = (f.order or []).index(f.default) if f.default in (f.order or []) else 0
                    values[f.name] = st.selectbox(
                        f.display, options=f.order, index=idx, key=f.name
                    )
    return values