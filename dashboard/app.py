"""
Project Risk Intelligence Platform - Streamlit Entry Point.
Run with: streamlit run dashboard/app.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

st.set_page_config(
    page_title="Project Risk Intelligence Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🛡️ Project Risk Intelligence Platform")
st.caption("Explainable ML for Four-Level Project-Risk Decision Support")

st.sidebar.markdown("## Navigation")
st.sidebar.markdown("Use the pages above to navigate the system.")
st.sidebar.info(
    "This DSS integrates supervised machine learning, SHAP-based explainable AI, "
    "and interactive analytics for project risk prediction."
)