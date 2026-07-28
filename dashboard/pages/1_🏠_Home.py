"""Home page - Project Risk Intelligence Platform overview."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st
from code.data_prep.load_data import load_raw
from code.data_prep.clean_data import clean
from code.utils.risk_levels import RISK_LEVELS, RISK_COLOURS, RISK_ACTIONS
from code.utils.config import RAW_DIR, ROOT

st.set_page_config(page_title="Home", page_icon="🏠", layout="wide")
st.title("🏠 Project Risk Intelligence Platform")
st.caption("A four-level ordinal risk decision-support system for project managers.")

# Load and clean data
raw_path = RAW_DIR / "project_risk_raw_dataset.csv"
if not raw_path.exists():
    raw_path = ROOT / "project_risk_raw_dataset.csv"
df = clean(load_raw(raw_path))

cols = st.columns(4)
cols[0].metric("Total Projects", f"{len(df):,}")
cols[1].metric("Distinct Risk Levels", df["Risk_Level"].nunique())
cols[2].metric("Mean Complexity", f"{df['Complexity_Score'].mean():.2f}")
cols[3].metric("Critical Share", f"{(df['Risk_Level'].eq('Critical').mean() * 100):.1f}%")

st.subheader("Risk Distribution")
counts = df["Risk_Level"].value_counts().reindex(RISK_LEVELS).fillna(0).astype(int)
st.bar_chart(counts)

st.subheader("Recommended Actions by Predicted Risk Level")
for k in RISK_LEVELS:
    st.markdown(
        f"<span style='background:{RISK_COLOURS[k]};color:white;padding:6px 10px;border-radius:6px;'>"
        f"<b>{k}</b></span> &nbsp; {RISK_ACTIONS[k]}",
        unsafe_allow_html=True,
    )

st.info("Use the sidebar to navigate to the Predictor, Analytics, and Model Performance pages.")